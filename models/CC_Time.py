import numpy as np
import torch
import torch.nn as nn
from torch import optim
from models.ReVIN import RevIN
from utils.CKA import CKA,CudaCKA
from custom_transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel, GPT2Tokenizer
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from utils.tools import positional_encoding
import torch.nn.functional as F
from layers.layer import Transpose, Bank_MultiheadAttention, _MultiheadAttention, TSTEncoderLayer



class CC_Time(nn.Module):
    def __init__(self, configs):
        super(CC_Time, self).__init__()
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.pred_len = configs.pred_len
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        if self.stride > 1 or self.patch_size > 1:
            self.patch_num += 1


        ##time series Branch
        self.n_heads = configs.n_heads
        self.ts_layer_num = configs.ts_layer_num
        self.ts_d_model = configs.ts_d_model
        self.ts_d_ff = configs.ts_d_ff
        self.patch_in_layer = nn.Linear(configs.patch_size, self.ts_d_model)
        self.ts_branch_layers = nn.ModuleList(
            [TSTEncoderLayer(self.patch_num, self.ts_d_model, n_heads=self.n_heads, d_k=None, d_v=None,
                             d_ff=self.ts_d_ff, norm='BatchNorm',
                             attn_dropout=0.1, dropout=0.2, res_attention=False, pre_norm=False, store_attn=False)
             for i in range(self.ts_layer_num)])

        self.revin_layer = RevIN(1, affine=True, subtract_last=False)
        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=self.patch_num, d_model=self.ts_d_model)


        ##LLM Branch
        self.llm_d_model = configs.llm_d_model
        self.llm_d_ff = configs.llm_d_ff
        self.gpt_layers = configs.gpt_layers
        self.llm_in_layer = nn.Linear(configs.patch_size, configs.llm_d_model)
        self.llm_in_layer2 = nn.Linear(configs.seq_len, configs.llm_d_model)
        self.llm_temporal_layers = GPT2Model.from_pretrained('./gpt2', output_attentions=True,
                                              output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.llm_temporal_layers.h = self.llm_temporal_layers.h[:configs.gpt_layers]

        self.llm_variable_layers = GPT2Model.from_pretrained('./gpt2', output_attentions=True,
                                              output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.llm_variable_layers.h = self.llm_variable_layers.h[:configs.gpt_layers]
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            './gpt2',
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        self.proj_layer1 = nn.Linear(configs.llm_d_model, self.llm_d_ff)
        self.proj_layer2 = nn.Linear(self.ts_d_model, configs.llm_d_model)
        self.dropout = nn.Dropout(0.2)

        self.dataset_name = configs.dataset_name
        variable_description_list = {"ETTm1":63, "ETTm2":63, "ETTh1":63, "ETTh2":63, "weather":61, "ZafNoo":75, "CzeLan":68, "electricity":32, "traffic":35}
        self.text_len_Linear = nn.Linear(variable_description_list[self.dataset_name], 1)

        self.qurey = nn.Parameter(torch.randn(configs.enc_in * configs.batch_size, self.ts_d_model, 1),
                                  requires_grad=True)
        self.qurey_linear = nn.Linear(1, self.patch_num)
        self.bias1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias2 = nn.Parameter(1 - self.bias1.data)

        self.variable_num = configs.variable_num
        self.left_linear_layer = nn.Linear(1, self.variable_num)
        self.right_linear_layer = nn.Linear(1, self.variable_num)
        self.bank_dim = configs.bank_dim
        self.alpha = configs.alpha
        self.extractor = nn.Parameter(torch.randn(configs.enc_in, self.bank_dim), requires_grad=True)
        self.extractor_linear = nn.Linear(configs.llm_d_model, self.bank_dim)
        self.extractor_attention = Bank_MultiheadAttention(self.bank_dim, 1, 64, 64, attn_dropout=0.1, proj_dropout=0.1,
                                                      res_attention=False)


        self.llm_to_ts = nn.Linear(configs.llm_d_model, self.ts_d_model)
        self.fusion_attention = nn.ModuleList([_MultiheadAttention(self.ts_d_model, self.n_heads, self.ts_d_model // self.n_heads,
                                                           self.ts_d_model // self.n_heads, attn_dropout=0.1, proj_dropout=0.1,
                                                           res_attention=False) for i in range(2)])
        self.current_memory_attention = _MultiheadAttention(self.ts_d_model, self.n_heads, self.ts_d_model // self.n_heads, self.ts_d_model // self.n_heads, attn_dropout=0.1, proj_dropout=0.1, res_attention=False)
        if configs.cross_norm == "BatchNorm":
            self.fusion_norm = nn.ModuleList([nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.ts_d_model), Transpose(1,2)) for i in range(self.ts_layer_num)])
        else:
            self.fusion_norm = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.ts_d_model)) for i in range(self.ts_layer_num)])


        self.llm_out_layer = nn.Linear(self.llm_d_ff * self.patch_num, configs.pred_len)
        self.ts_out_layer = nn.Linear(self.ts_d_model*self.patch_num, configs.pred_len)
        self.llm_variable_out_layer = nn.Linear(self.llm_d_ff, configs.pred_len)


        for i, (name, param) in enumerate(self.llm_temporal_layers.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            elif 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for i, (name, param) in enumerate(self.llm_variable_layers.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            elif 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False




    def forward(self, x, y, *args, **kwargs):
        B, L, M = x.shape
        output_dic = {}
        x = self.revin_layer(x, 'norm')
        x = rearrange(x, 'b l m -> b m l')
        llm_variable_embedding = self.llm_in_layer2(x)


        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        llm_temporal_embedding = self.llm_in_layer(x)
        x = rearrange(x, 'b m n p -> (b m) n p')



        if self.dataset_name == "weather":
            prompt = ["p (mbar): Atmospheric pressure measured in millibars. Atmospheric pressure is an important parameter in meteorology, influencing weather and climate changes. The statistics of p (mbar): minimum:962.08, maximum:1020.07, mean:990.64, variance:8.63.",
                      "T (degC): Temperature is a key parameter that describes climate conditions. It is closely related to humidity. As the temperature rises, the air can hold more moisture, which in turn affects humidity. The statistics of T (degC): minimum:-6.44, maximum:34.8, mean:11.98, variance:7.72.",
                      "pot (K): Potential temperature, the temperature an air parcel would have if it were expanded or compressed adiabatically to a standard pressure (usually 1000 hPa), measured in Kelvin. Potential temperature is used to study the thermal properties and stability of air parcels. The statistics of Tpot (K): minimum:266.19, maximum:309.13, mean:285.91, variance:7.92.",
                      "Tdew (degC): Dew point temperature, the temperature at which air becomes saturated with moisture and dew can form, measured in degrees Celsius. Dew point temperature is used to measure the humidity of the air. The statistics of Tdew (degC): minimum:-13.81, maximum:20.5, mean:5.47, variance:6.46.",
                      "rh (%): Relative humidity, representing the percentage of moisture in the air relative to the maximum amount the air can hold at that temperature. Relative humidity describes the moisture content of the air. The statistics of rh (%): minimum:21.16, maximum:100.0, mean:67.53, variance:18.76.",
                      "VPmax (mbar): Maximum vapor pressure, the pressure at which water vapor is in equilibrium with its liquid or solid form at the current temperature, measured in millibars. Maximum vapor pressure is an important parameter for calculating relative humidity and dew point temperature. The statistics of VPmax (mbar): minimum:3.77, maximum:55.67, mean:15.68, variance:8.11.",
                      "VPact (mbar): Actual vapor pressure, the partial pressure of water vapor in the air, measured in millibars. Actual vapor pressure describes the actual amount of water vapor in the air. The statistics of VPact (mbar): minimum:2.09, maximum:24.16, mean:9.84, variance:4.4.",
                      "VPdef (mbar): Vapor pressure deficit, the difference between the maximum vapor pressure and the actual vapor pressure, measured in millibars. Vapor pressure deficit measures the drying power of the air. The statistics of VPdef (mbar): minimum:0.0, maximum:42.1, mean:5.84, variance:5.9.",
                      "sh (g/kg): Specific humidity, the mass of water vapor per unit mass of air, measured in grams per kilogram. Specific humidity describes the moisture characteristics of the air. The statistics of sh (g/kg): minimum:1.3, maximum:15.4, mean:6.22, variance:2.8.",
                      "H2OC (mmol/mol): Water vapor concentration, the molar concentration of water vapor in the air, measured in millimoles per mole. Water vapor concentration is used to study the amount of water vapor in the air. The statistics of H2OC (mmol/mol): minimum:2.09, maximum:24.53, mean:9.95, variance:4.46.",
                      "rho (g/m³): Air density, the mass of air per unit volume, measured in grams per cubic meter. Air density describes the density of the air and affects the dynamics of the atmosphere. The statistics of rho (g/m**3): minimum:1107.38, maximum:1318.52, mean:1206.75, variance:38.07.",
                      "wv (m/s): Wind velocity, the speed of the wind, measured in meters per second. Wind velocity describes the strength of the wind and its impact on meteorological phenomena. The statistics of wv (m/s): minimum:-9999.0, maximum:13.77, mean:1.95, variance:52.1.",
                      "max. wv (m/s): Maximum wind velocity, the highest wind speed recorded, measured in meters per second. Maximum wind velocity describes extreme wind speed events and their impact. The statistics of max. wv (m/s): minimum:0.0, maximum:22.9, mean:3.76, variance:2.6.",
                      "wd (deg): Wind direction, the direction from which the wind is blowing, measured in degrees. Wind direction describes the direction and path of the wind. The statistics of wd (deg): minimum:0.0, maximum:360.0, mean:175.99, variance:86.97.",
                      "rain (mm): Amount of rainfall, the depth of rain that falls over a given area, measured in millimeters. Rainfall amount describes the intensity and total amount of precipitation. The statistics of rain (mm): minimum:0.0, maximum:11.2, mean:0.01, variance:0.12.",
                      "raining (s): Duration of rainfall, the total time it rained, measured in seconds. Rainfall duration is used to study the temporal characteristics of precipitation events. The statistics of raining (s): minimum:0.0, maximum:600.0, mean:23.29, variance:105.35.",
                      "SWDR (W/m²): Shortwave downward radiation, the amount of solar radiation reaching the Earth's surface, measured in watts per square meter. Shortwave radiation describes the input of solar energy. The statistics of SWDR: minimum:0.0, maximum:1115.29, mean:160.64, variance:238.78.",
                      "PAR (µmol/m²/s): Photosynthetically active radiation, the portion of light (400-700 nm) usable by plants for photosynthesis, measured in micromoles per square meter per second. Photosynthetically active radiation is used to study conditions for plant growth and photosynthesis. The statistics of PAR: minimum:0.0, maximum:2131.76, mean:318.04, variance:468.06.",
                      "max. PAR (µmol/m²/s): Maximum photosynthetically active radiation, the highest value of PAR recorded, measured in micromoles per square meter per second. Maximum photosynthetically active radiation is used to study extreme light conditions and their impact. The statistics of max. PAR: minimum:-9999.0, maximum:2498.94, mean:378.48, variance:643.36.",
                      "Tlog (degC): Logged temperature, a recorded value of temperature, measured in degrees Celsius. Logged temperature is used for historical temperature data preservation and analysis. The statistics of Tlog (degC): minimum:6.96, maximum:49.09, mean:22.81, variance:7.98.",
                      "OT: Target variable, which could be temperature or another specific meteorological parameter that the prediction model aims to forecast. The target variable is the object of prediction for the time series prediction model. The statistics of OT: minimum:-9999.0, maximum:524.2, mean:411.05, variance:383.96."]

        elif self.dataset_name == "CzeLan":
            prompt =["CZE_LAN_Cbe_Jt_16: This variable likely represents sap flow measurements. It could be in units of sap flow rate, such as grams per hour (g/h) or liters per hour (L/h), indicating the volume of sap flowing through a given area of the plant over a specific period. The statistics of CZE_LAN_Cbe_Jt_16: minimum:0.0, maximum:16019.29, mean:3523.85, variance:4296.13.",
                     "ta: Air temperature, typically measured in degrees Celsius (°C). This variable reflects the ambient temperature surrounding the plants. The statistics of ta: minimum:3.6, maximum:33.34, mean:19.29, variance:5.2.",
                     "rh: Relative humidity, usually expressed as a percentage (%). It indicates the amount of moisture in the air relative to the maximum amount of moisture the air can hold at a given temperature. The statistics of rh: minimum:23.46, maximum:100.0, mean:70.77, variance:21.31.",
                     "sw_in: Shortwave incoming radiation, usually measured in watts per square meter (W/m²). It quantifies the amount of solar radiation reaching the surface.The statistics of sw_in: minimum:0.0, maximum:1146.78, mean:251.02, variance:300.55. ",
                     "ppfd_in: Photosynthetic photon flux density incoming, typically measured in micromoles per square meter per second (µmol/m²/s). This measures the amount of light available for photosynthesis. The statistics of ppfd_in: minimum:0.0, maximum:2424.29, mean:530.66, variance:635.36.",
                     "ws: Wind speed, usually measured in meters per second (m/s) or kilometers per hour (km/h). This variable indicates the speed of the wind affecting the plants. The statistics of ws: minimum:0.12, maximum:5.85, mean:1.55, variance:0.79.",
                     "precip: Precipitation, measured in millimeters (mm). This variable quantifies the amount of rainfall received over a certain period. The statistics of precip: minimum:0.0, maximum:14.4, mean:0.02, variance:0.25.",
                     "swc_shallow: Shallow soil water content, typically measured in volumetric percentage (%). It represents the amount of water present in the shallow soil layer. The statistics of swc_shallow: minimum:0.32, maximum:0.48, mean:0.39, variance:0.04.",
                     "swc_deep: Deep soil water content, also measured in volumetric percentage (%). It indicates the amount of water in the deeper soil layers. The statistics of swc_deep: minimum:0.4, maximum:0.45, mean:0.42, variance:0.01.",
                     "ext_rad: Extraterrestrial radiation, usually measured in watts per square meter (W/m²). It quantifies the solar radiation received at the top of the Earth's atmosphere. The statistics of ext_rad: minimum:0.0, maximum:1196.14, mean:463.17, variance:454.78.",
                     "OT (vpd): Vapor pressure deficit, typically measured in kilopascals (kPa). It represents the difference between the amount of moisture in the air and how much moisture the air can hold when it is saturated. This variable is crucial for understanding plant transpiration rates. The statistics of OT: minimum:0.0, maximum:3.17, mean:0.79, variance:0.71."]


        elif self.dataset_name == "ZafNoo":
            prompt = ["ZAF_NOO_E3_IRR_Mdo_Jt_2: This variable likely represents sap flow measurements. It could be in units of sap flow rate, such as grams per hour (g/h) or liters per hour (L/h), indicating the volume of sap flowing through a given area of the plant over a specific period. The statistics of ZAF_NOO_E3_IRR_Mdo_Jt_2: minimum:0.0, maximum:9656.96, mean:410.01, variance:877.68.",
                      "ta: Air temperature, typically measured in degrees Celsius (°C). This variable reflects the ambient temperature surrounding the plants. The statistics of ta: minimum:-1.16, maximum:33.93, mean:12.85, variance:6.52.",
                      "rh: Relative humidity, usually expressed as a percentage (%). It indicates the amount of moisture in the air relative to the maximum amount of moisture the air can hold at a given temperature. The statistics of rh: minimum:6.03, maximum:100.0, mean:61.77, variance:21.93.",
                      "sw_in: Shortwave incoming radiation, usually measured in watts per square meter (W/m²). It quantifies the amount of solar radiation reaching the surface. The statistics of sw_in: minimum:0.0, maximum:1179.0, mean:215.04, variance:315.14.",
                      "ppfd_in: Photosynthetic photon flux density incoming, typically measured in micromoles per square meter per second (µmol/m²/s). This measures the amount of light available for photosynthesis. The statistics of ppfd_in: minimum:0.0, maximum:2492.41, mean:454.59, variance:666.21.",
                      "ws: Wind speed, usually measured in meters per second (m/s) or kilometers per hour (km/h). This variable indicates the speed of the wind affecting the plants. The statistics of ws: minimum:0.0, maximum:11.11, mean:2.5, variance:1.34.",
                      "precip: Precipitation, measured in millimeters (mm). This variable quantifies the amount of rainfall received over a certain period. The statistics of precip: minimum:0.0, maximum:20.07, mean:0.12, variance:0.67.",
                      "swc_shallow: Shallow soil water content, typically measured in volumetric percentage (%). It represents the amount of water present in the shallow soil layer. The statistics of swc_shallow: minimum:0.08, maximum:0.34, mean:0.18, variance:0.04.",
                      "swc_deep: Deep soil water content, also measured in volumetric percentage (%). It indicates the amount of water in the deeper soil layers. The statistics of swc_deep: minimum:0.08, maximum:0.3, mean:0.1, variance:0.01.",
                      "ext_rad: Extraterrestrial radiation, usually measured in watts per square meter (W/m²). It quantifies the solar radiation received at the top of the Earth's atmosphere. The statistics of ext_rad: minimum:0.0, maximum:1391.89, mean:339.29, variance:434.28.",
                      "OT (vpd): Vapor pressure deficit, typically measured in kilopascals (kPa). It represents the difference between the amount of moisture in the air and how much moisture the air can hold when it is saturated. This variable is crucial for understanding plant transpiration rates. The statistics of OT: minimum:0.0, maximum:4.4, mean:0.73, variance:0.7."]

        elif "ETT" in self.dataset_name:
            prompt = [
                "HUFL (High UseFul Load): Represents the useful power or load of the system or equipment under high load conditions during the recorded time period. The statistics of HUFL: minimum:-18.75, maximum:23.64, mean:7.80, variance:6.13.",
                "HULL (High UseLess Load): Represents the useless power or load of the system or equipment under high load conditions during the recorded time period. The statistics of HULL: minimum:-4.75, maximum:10.11, mean:1.96, variance:2.14.",
                "MUFL (Middle UseFul Load): Represents the useful power or load of the system or equipment under medium load conditions during the recorded time period. The statistics of MUFL: minimum:-21.28, maximum:17,34, mean:4.85, variance:5.90.",
                "MULL (Middle UseLess Load): Represents the useless power or load of the system or equipment under medium load conditions during the recorded time period. The statistics of MULL: minimum:-5.93, maximum:7.56, mean:0.702, variance:1.97.",
                "LUFL (Low UseFul Load): Represents the useful power or load of the system or equipment under low load conditions during the recorded time period. The statistics of LUFL: minimum:-1.18, maximum:8.49, mean:2.99, variance:1.25.",
                "LULL (Low UseLess Load): Represents the useless power or load of the system or equipment under low load conditions during the recorded time period. The statistics of LULL: minimum:-1.37, maximum:3.04, mean:0.77, variance:0.66.",
                "OT (Oil Temperature): The target variable representing the oil temperature during the recorded time period. The goal of the prediction model is to predict this variable based on other variables. The statistics of OT: minimum:-4.08, maximum:46.00, mean:17.29, variance:8.51."]

        else:
            prompt = None




        if prompt == None:
            llm_x = llm_variable_embedding
        else:
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
            prompt_embeddings = self.llm_temporal_layers.get_input_embeddings()(prompt.to(x.device))  # (variable_num, prompt_token, dim)
            prompt_embeddings = self.text_len_Linear(prompt_embeddings.permute(0,2,1)).permute(0,2,1).squeeze(dim=1)
            llm_x = prompt_embeddings + llm_variable_embedding


        extractor_llm_x = self.extractor_linear(llm_x)

        bank_output, bank_weights = self.extractor_attention(extractor_llm_x, self.extractor, self.extractor) # bank_weights:[bs, 1, num_nodes, num_nodes]


        llm_temporal_outputs = rearrange(llm_temporal_embedding, 'b m n p -> (b m) n p')
        llm_temporal_hiddens = self.llm_temporal_layers(inputs_embeds=llm_temporal_outputs).hidden_states
        llm_temporal_outputs = llm_temporal_hiddens[-1]
        llm_temporal_outputs = self.proj_layer1(llm_temporal_outputs)
        llm_temporal_outputs = self.dropout(llm_temporal_outputs)


        llm_variable_outputs = llm_x
        llm_variable_hiddens = self.llm_variable_layers(inputs_embeds=llm_variable_outputs, alpha=self.alpha, correlation_attn_weights=bank_weights).hidden_states
        llm_variable_outputs = llm_variable_hiddens[-1]


        ts_outputs = self.patch_in_layer(x) + self.W_pos

        llm_variable_hiddens_1 = rearrange(llm_variable_hiddens[1], 'b m p -> (b m) 1 p')
        llm_variable_hiddens_1 = llm_variable_hiddens_1.repeat(1, self.variable_num, 1)
        llm_memory_hidden = torch.cat([llm_variable_hiddens_1, llm_temporal_hiddens[1], llm_variable_hiddens_1], dim=1)
        llm_memory_hidden = self.llm_to_ts(llm_memory_hidden)



        for i in range(self.ts_layer_num):
            if i == 0:
                llm_variable_hiddens_i = llm_variable_hiddens[2*i+1]
                llm_variable_hiddens_i = rearrange(llm_variable_hiddens_i, 'b m p -> (b m) 1 p')
                llm_variable_hiddens_i = llm_variable_hiddens_i.repeat(1, self.variable_num, 1)
                llm_current_hidden = torch.cat([llm_variable_hiddens_i, llm_temporal_hiddens[2 * i + 1], llm_variable_hiddens_i], dim=1)
                llm_current_hidden = self.llm_to_ts(llm_current_hidden)
                llm_ts_fusion_outputs = self.fusion_attention[0](ts_outputs, llm_current_hidden, llm_current_hidden)[0]
                llm_ts_fusion_outputs = self.fusion_norm[i](llm_ts_fusion_outputs)

                ts_outputs = self.ts_branch_layers[i](ts_outputs, ts_outputs)
                ts_outputs = llm_ts_fusion_outputs+ts_outputs

            else:
                llm_variable_hiddens_i = llm_variable_hiddens[2*i+1]
                llm_variable_hiddens_i = rearrange(llm_variable_hiddens_i, 'b m p -> (b m) 1 p')
                llm_variable_hiddens_i = llm_variable_hiddens_i.repeat(1, self.variable_num, 1)
                llm_current_hidden = torch.cat([llm_variable_hiddens_i, llm_temporal_hiddens[2 * i + 1], llm_variable_hiddens_i], dim=1)
                llm_current_hidden = self.llm_to_ts(llm_current_hidden)
                self.bias1.to(llm_current_hidden.device)
                self.bias2.to(llm_current_hidden.device)
                llm_current_hidden = (self.bias1*self.current_memory_attention(llm_current_hidden, llm_memory_hidden, llm_memory_hidden)[0] +
                              self.bias2*self.current_memory_attention(llm_memory_hidden, llm_current_hidden, llm_current_hidden)[0])


                llm_ts_fusion_outputs = self.fusion_attention[0](ts_outputs, llm_current_hidden, llm_current_hidden)[0]
                llm_ts_fusion_outputs = self.fusion_norm[i](llm_ts_fusion_outputs)

                ts_outputs = self.ts_branch_layers[i](ts_outputs, ts_outputs)
                ts_outputs = llm_ts_fusion_outputs+ts_outputs

                llm_memory_hidden = llm_current_hidden


        llm_outputs = llm_temporal_outputs
        llm_outputs = self.llm_out_layer(llm_outputs.reshape(B * M, -1))
        llm_outputs = rearrange(llm_outputs, '(b m) l -> b l m', b=B) + self.llm_variable_out_layer(llm_variable_outputs).permute(0,2,1)
        llm_outputs = self.revin_layer(llm_outputs, 'denorm')



        ts_outputs = self.ts_out_layer(ts_outputs.reshape(B * M, -1))
        ts_outputs = rearrange(ts_outputs, '(b m) l -> b l m', b=B)
        ts_outputs = self.revin_layer(ts_outputs, 'denorm')

        output_dic["llm_outputs"] = llm_outputs
        output_dic["ts_outputs"] = ts_outputs
        return output_dic

