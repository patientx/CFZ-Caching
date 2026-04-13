from .cfz_caching_condition import save_conditioning, load_conditioning, CFZ_PrintMarker, CFZ_CUDNN, CFZ_CUDNN_Advanced
from .cfz_miopen import CFZ_MIOpen_Profile, CFZ_MIOpen_Settings, CFZ_MIOpen_Solvers, CFZ_MIOpen_Paths, CFZ_MIOpen_DBInfo

NODE_CLASS_MAPPINGS = {
    "CFZ_save_conditioning": save_conditioning,
    "CFZ_load_conditioning": load_conditioning,
    "CFZ_PrintMarker": CFZ_PrintMarker,
    "CFZ_CUDNN": CFZ_CUDNN,
    "CFZ_CUDNN_Advanced": CFZ_CUDNN_Advanced,
    "CFZ_MIOpen_Profile": CFZ_MIOpen_Profile,
    "CFZ_MIOpen_Settings": CFZ_MIOpen_Settings,
    "CFZ_MIOpen_Solvers": CFZ_MIOpen_Solvers,
    "CFZ_MIOpen_Paths": CFZ_MIOpen_Paths,
    "CFZ_MIOpen_DBInfo": CFZ_MIOpen_DBInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFZ_save_conditioning": "CFZ Save Conditioning",
    "CFZ_load_conditioning": "CFZ Load Conditioning",
    "CFZ_PrintMarker": "CFZ Print Marker",
    "CFZ_CUDNN": "CFZ CUDNN",
    "CFZ_CUDNN_Advanced": "CFZ CUDNN Advanced",
    "CFZ_MIOpen_Profile": "CFZ MIOpen Profile",
    "CFZ_MIOpen_Settings": "CFZ MIOpen Settings",
    "CFZ_MIOpen_Solvers": "CFZ MIOpen Solvers",
    "CFZ_MIOpen_Paths": "CFZ MIOpen Paths",
    "CFZ_MIOpen_DBInfo": "CFZ MIOpen DB Info",
}