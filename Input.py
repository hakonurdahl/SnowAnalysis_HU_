input_data = {

    #Period

    "tot": {"period": (1960, 2024), "scenario": None, "title":"1960-2024"},
    "new": {"period": (1991, 2024),"scenario": None, "title":"1991-2024"},
    "old": {"period": (1960, 1990),"scenario": None, "title":"1960-1990"},
    "future_rcp45": {"period": (2024, 2074),"scenario": "rcp45", "title":"2024-2074 RCP 4.5"},
    "future_rcp85": {"period": (2024, 2074),"scenario": "rcp85", "title":"2024-2074 RCP 8.5"},
    "new_old": {"period": ("", ""),"scenario": None, "title":""},
    

    #Variable
    "beta": {"limits": (2,6),"label": "$\\beta$", "title": "Reliability Index per Municipality\n"},
    "opt_beta": {"limits": (2,6),"label": "$\\beta$", "title": "Reliability Index per Municipality\n"},
    "char": {"limits": (0, 6),"label": "$s_{k} [kN/m]$", "title": "Characteristic Value per Municipality\n"},
    "cov": {"limits": (0.3,0.8),"label": "CoV", "title": "Coefficient of Variance per Municipality\n"},
    "opt_char": {"limits": (0,6),"label": '$s_{k,opt} [kN/m]$', "title": "Optimal Characteristic Values per Municipality\n"},
    "diff_beta": {"limits": (-2,2),"label": "$\\Delta\\beta$", "title": "Change in Reliability Index per Municipality"},
    "T50_char": {"limits": (0, 6),"label": "$s_{k, T_{50}} [kN/m]$", "title": "Characteristic Value per Municipality, $T_{50}$\n"},
    "T50_beta": {"limits": (2,6),"label": "$\\beta$", "title": "Reliability Index per Municipality\n"},
}