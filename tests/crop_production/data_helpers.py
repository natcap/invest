"""Helper module for generating test data related to Crop Production tests."""
import pandas


def sample_nutrient_df():
    """Generate a sample nutrient DataFrame for crops.

    This function creates a small DataFrame containing example nutrient
    and production data for crops such as corn and soybean. It can be
    used for testing nutrient calculations and validating model outputs.

    Returns:
        pandas.DataFrame: Nutrient and production data indexed by crop name.

    """

    return pandas.DataFrame([
        {'crop_name': 'corn', 'area (ha)': 21.0, 'production_observed': 0.2,
         'percentrefuse': 7, 'protein': 42., 'lipid': 8, 'energy': 476.,
         'ca': 27.0, 'fe': 15.7, 'mg': 280.0, 'ph': 704.0, 'k': 1727.0,
         'na': 2.0, 'zn': 4.9, 'cu': 1.9, 'fl': 8, 'mn': 2.9, 'se': 0.1,
         'vita': 3.0, 'betac': 16.0, 'alphac': 2.30, 'vite': 0.8,
         'crypto': 1.6, 'lycopene': 0.36, 'lutein': 63.0, 'betat': 0.5,
         'gammat': 2.1, 'deltat': 1.9, 'vitc': 6.8, 'thiamin': 0.4,
         'riboflavin': 1.8, 'niacin': 8.2, 'pantothenic': 0.9,
         'vitb6': 1.4, 'folate': 385.0, 'vitb12': 2.0, 'vitk': 41.0},

        {'crop_name': 'soybean', 'area (ha)': 5., 'production_observed': 4.,
         'percentrefuse': 9, 'protein': 33., 'lipid': 2., 'energy': 99.,
         'ca': 257., 'fe': 15.7, 'mg': 280., 'ph': 704.0, 'k': 197.0,
         'na': 2., 'zn': 4.9, 'cu': 1.6, 'fl': 3., 'mn': 5.2, 'se': 0.3,
         'vita': 3.0, 'betac': 16.0, 'alphac': 1.0, 'vite': 0.8,
         'crypto': 0.6, 'lycopene': 0.3, 'lutein': 61.0, 'betat': 0.5,
         'gammat': 2.3, 'deltat': 1.2, 'vitc': 3.0, 'thiamin': 0.42,
         'riboflavin': 0.82, 'niacin': 12.2, 'pantothenic': 0.92,
         'vitb6': 5.4, 'folate': 305., 'vitb12': 3., 'vitk': 42.},
         ]).set_index('crop_name')


def tabulate_regr_results_table():
    """Generate expected output for unit tests of tabulated regression results.

    This function returns a DataFrame that represents the expected output
    of the `tabulate_regression_results` function, with the parameters
    defined in the unit test `test_tabulate_regression_results`.

    Returns:
        pandas.DataFrame: Expected tabulated results for regression output.

    """

    return pandas.DataFrame([
        {'crop_name': 'corn', 'area (ha)': 20.0,
         'production_observed': 80.0, 'production_modeled': 40.0,
         'protein_modeled': 15624000.0, 'protein_observed': 31248000.0,
         'lipid_modeled': 2976000.0, 'lipid_observed': 5952000.0,
         'energy_modeled': 177072000.0, 'energy_observed': 354144000.0,
         'ca_modeled': 10044000.0, 'ca_observed': 20088000.0,
         'fe_modeled': 5840400.0, 'fe_observed': 11680800.0,
         'mg_modeled': 104160000.0, 'mg_observed': 208320000.0,
         'ph_modeled': 261888000.0, 'ph_observed': 523776000.0,
         'k_modeled': 642444000.0, 'k_observed': 1284888000.0,
         'na_modeled': 744000.0, 'na_observed': 1488000.0,
         'zn_modeled': 1822800.0, 'zn_observed': 3645600.0,
         'cu_modeled': 706800.0, 'cu_observed': 1413600.0,
         'fl_modeled': 2976000.0, 'fl_observed': 5952000.0,
         'mn_modeled': 1078800.0, 'mn_observed': 2157600.0,
         'se_modeled': 37200.0, 'se_observed': 74400.0,
         'vita_modeled': 1116000.0, 'vita_observed': 2232000.0,
         'betac_modeled': 5952000.0, 'betac_observed': 11904000.0,
         'alphac_modeled': 855600.0, 'alphac_observed': 1711200.0,
         'vite_modeled': 297600.0, 'vite_observed': 595200.0,
         'crypto_modeled': 595200.0, 'crypto_observed': 1190400.0,
         'lycopene_modeled': 133920.0, 'lycopene_observed': 267840.0,
         'lutein_modeled': 23436000.0, 'lutein_observed': 46872000.0,
         'betat_modeled': 186000.0, 'betat_observed': 372000.0,
         'gammat_modeled': 781200.0, 'gammat_observed': 1562400.0,
         'deltat_modeled': 706800.0, 'deltat_observed': 1413600.0,
         'vitc_modeled': 2529600.0, 'vitc_observed': 5059200.0,
         'thiamin_modeled': 148800.0, 'thiamin_observed': 297600.0,
         'riboflavin_modeled': 669600.0, 'riboflavin_observed': 1339200.0,
         'niacin_modeled': 3050400.0, 'niacin_observed': 6100800.0,
         'pantothenic_modeled': 334800.0, 'pantothenic_observed': 669600.0,
         'vitb6_modeled': 520800.0, 'vitb6_observed': 1041600.0,
         'folate_modeled': 143220000.0, 'folate_observed': 286440000.0,
         'vitb12_modeled': 744000.0, 'vitb12_observed': 1488000.0,
         'vitk_modeled': 15252000.0, 'vitk_observed': 30504000.0},
        {'crop_name': 'soybean', 'area (ha)': 40.0,
         'production_observed': 120.0, 'production_modeled': 70.0,
         'protein_modeled': 21021000.0, 'protein_observed': 36036000.0,
         'lipid_modeled': 1274000.0, 'lipid_observed': 2184000.0,
         'energy_modeled': 63063000.0, 'energy_observed': 108108000.0,
         'ca_modeled': 163709000.0, 'ca_observed': 280644000.0,
         'fe_modeled': 10000900.0, 'fe_observed': 17144400.0,
         'mg_modeled': 178360000.0, 'mg_observed': 305760000.0,
         'ph_modeled': 448448000.0, 'ph_observed': 768768000.0,
         'k_modeled': 125489000.0, 'k_observed': 215124000.0,
         'na_modeled': 1274000.0, 'na_observed': 2184000.0,
         'zn_modeled': 3121300.0, 'zn_observed': 5350800.0,
         'cu_modeled': 1019200.0, 'cu_observed': 1747200.0,
         'fl_modeled': 1911000.0, 'fl_observed': 3276000.0,
         'mn_modeled': 3312400.0, 'mn_observed': 5678400.0,
         'se_modeled': 191100.0, 'se_observed': 327600.0,
         'vita_modeled': 1911000.0, 'vita_observed': 3276000.0,
         'betac_modeled': 10192000.0, 'betac_observed': 17472000.0,
         'alphac_modeled': 637000.0, 'alphac_observed': 1092000.0,
         'vite_modeled': 509600.0, 'vite_observed': 873600.0,
         'crypto_modeled': 382200.0, 'crypto_observed': 655200.0,
         'lycopene_modeled': 191100.0, 'lycopene_observed': 327600.0,
         'lutein_modeled': 38857000.0, 'lutein_observed': 66612000.0,
         'betat_modeled': 318500.0, 'betat_observed': 546000.0,
         'gammat_modeled': 1465100.0, 'gammat_observed': 2511600.0,
         'deltat_modeled': 764400.0, 'deltat_observed': 1310400.0,
         'vitc_modeled': 1911000.0, 'vitc_observed': 3276000.0,
         'thiamin_modeled': 267540.0, 'thiamin_observed': 458640.0,
         'riboflavin_modeled': 522340.0, 'riboflavin_observed': 895440.0,
         'niacin_modeled': 7771400.0, 'niacin_observed': 13322400.0,
         'pantothenic_modeled': 586040.0, 'pantothenic_observed': 1004640.0,
         'vitb6_modeled': 3439800.0, 'vitb6_observed': 5896800.0,
         'folate_modeled': 194285000.0, 'folate_observed': 333060000.0,
         'vitb12_modeled': 1911000.0, 'vitb12_observed': 3276000.0,
         'vitk_modeled': 26754000.0, 'vitk_observed': 45864000.0}])


def tabulate_pctl_results_table():
    """Generate expected output for unit tests of tabulated percentile results.

    This function returns the expected DataFrame output of the
    `tabulate_percentile_results` function, with the parameters
    defined in the unit test `test_tabulate_percentile_results`.

    Returns:
        pandas.DataFrame: Expected tabulated results for percentile output.

    """

    return pandas.DataFrame({
        "crop_name": ["corn", "soybean"], "area (ha)": [2, 4],
        "production_observed": [4, 7], "production_25": [1.25, 2.25],
        "production_50": [2.5, 4.5], "production_75": [3.75, 6.75],
        "protein_25": [488250, 675675], "protein_50": [976500, 1351350],
        "protein_75": [1464750, 2027025],
        "protein_observed": [1562400, 2102100],
        "lipid_25": [93000, 40950], "lipid_50": [186000, 81900],
        "lipid_75": [279000, 122850], "lipid_observed": [297600, 127400],
        "energy_25": [5533500, 2027025], "energy_50": [11067000, 4054050],
        "energy_75": [16600500, 6081075],
        "energy_observed": [17707200, 6306300],
        "ca_25": [313875, 5262075], "ca_50": [627750, 10524150],
        "ca_75": [941625, 15786225], "ca_observed": [1004400, 16370900],
        "fe_25": [182512.5, 321457.5], "fe_50": [365025, 642915],
        "fe_75": [547537.5, 964372.5], "fe_observed": [584040, 1000090],
        "mg_25": [3255000, 5733000], "mg_50": [6510000, 11466000],
        "mg_75": [9765000, 17199000], "mg_observed": [10416000, 17836000],
        "ph_25": [8184000, 14414400], "ph_50": [16368000, 28828800],
        "ph_75": [24552000, 43243200], "ph_observed": [26188800, 44844800],
        "k_25": [20076375, 4033575], "k_50": [40152750, 8067150],
        "k_75": [60229125, 12100725], "k_observed": [64244400, 12548900],
        "na_25": [23250, 40950], "na_50": [46500, 81900],
        "na_75": [69750, 122850], "na_observed": [74400, 127400],
        "zn_25": [56962.5, 100327.5], "zn_50": [113925, 200655],
        "zn_75": [170887.5, 300982.5], "zn_observed": [182280, 312130],
        "cu_25": [22087.5, 32760], "cu_50": [44175, 65520],
        "cu_75": [66262.5, 98280], "cu_observed": [70680, 101920],
        "fl_25": [93000, 61425], "fl_50": [186000, 122850],
        "fl_75": [279000, 184275], "fl_observed": [297600, 191100],
        "mn_25": [33712.5, 106470], "mn_50": [67425, 212940],
        "mn_75": [101137.5, 319410], "mn_observed": [107880, 331240],
        "se_25": [1162.5, 6142.5], "se_50": [2325, 12285],
        "se_75": [3487.5, 18427.5], "se_observed": [3720, 19110],
        "vita_25": [34875, 61425], "vita_50": [69750, 122850],
        "vita_75": [104625, 184275], "vita_observed": [111600, 191100],
        "betac_25": [186000, 327600], "betac_50": [372000, 655200],
        "betac_75": [558000, 982800], "betac_observed": [595200, 1019200],
        "alphac_25": [26737.5, 20475], "alphac_50": [53475, 40950],
        "alphac_75": [80212.5, 61425], "alphac_observed": [85560, 63700],
        "vite_25": [9300, 16380], "vite_50": [18600, 32760],
        "vite_75": [27900, 49140], "vite_observed": [29760, 50960],
        "crypto_25": [18600, 12285], "crypto_50": [37200, 24570],
        "crypto_75": [55800, 36855], "crypto_observed": [59520, 38220],
        "lycopene_25": [4185, 6142.5], "lycopene_50": [8370, 12285],
        "lycopene_75": [12555, 18427.5], "lycopene_observed": [13392, 19110],
        "lutein_25": [732375, 1248975], "lutein_50": [1464750, 2497950],
        "lutein_75": [2197125, 3746925], "lutein_observed": [2343600, 3885700],
        "betat_25": [5812.5, 10237.5], "betat_50": [11625, 20475],
        "betat_75": [17437.5, 30712.5], "betat_observed": [18600, 31850],
        "gammat_25": [24412.5, 47092.5], "gammat_50": [48825, 94185],
        "gammat_75": [73237.5, 141277.5], "gammat_observed": [78120, 146510],
        "deltat_25": [22087.5, 24570], "deltat_50": [44175, 49140],
        "deltat_75": [66262.5, 73710], "deltat_observed": [70680, 76440],
        "vitc_25": [79050, 61425], "vitc_50": [158100, 122850],
        "vitc_75": [237150, 184275], "vitc_observed": [252960, 191100],
        "thiamin_25": [4650, 8599.5], "thiamin_50": [9300, 17199],
        "thiamin_75": [13950, 25798.5], "thiamin_observed": [14880, 26754],
        "riboflavin_25": [20925, 16789.5], "riboflavin_50": [41850, 33579],
        "riboflavin_75": [62775, 50368.5],
        "riboflavin_observed": [66960, 52234], "niacin_25": [95325, 249795],
        "niacin_50": [190650, 499590], "niacin_75": [285975, 749385],
        "niacin_observed": [305040, 777140],
        "pantothenic_25": [10462.5, 18837], "pantothenic_50": [20925, 37674],
        "pantothenic_75": [31387.5, 56511],
        "pantothenic_observed": [33480, 58604],
        "vitb6_25": [16275, 110565], "vitb6_50": [32550, 221130],
        "vitb6_75": [48825, 331695], "vitb6_observed": [52080, 343980],
        "folate_25": [4475625, 6244875], "folate_50": [8951250, 12489750],
        "folate_75": [13426875, 18734625],
        "folate_observed": [14322000, 19428500],
        "vitb12_25": [23250, 61425], "vitb12_50": [46500, 122850],
        "vitb12_75": [69750, 184275], "vitb12_observed": [74400, 191100],
        "vitk_25": [476625, 859950], "vitk_50": [953250, 1719900],
        "vitk_75": [1429875, 2579850], "vitk_observed": [1525200, 2675400]
    })


def aggregate_regr_polygons_table():
    """Generate expected aggregated nutrient results by polygon (e.g., FID).

    This function returns the expected DataFrame output of the
    `aggregate_regression_results_to_polygons` function for use in the
    `test_aggregate_regression_results_to_polygons` unit test. It
    summarizes nutrient production values for multiple crops grouped by
    polygon identifiers (FIDs).

    Returns:
        pandas.DataFrame: Aggregated modeled and observed values by FID.

    """

    return pandas.DataFrame([
        {"FID": 0, "corn_modeled": 10, "corn_observed": 40,
         "soybean_modeled": 20, "soybean_observed": 50,
         "protein_modeled": 9912000, "protein_observed": 30639000,
         "lipid_modeled": 1108000, "lipid_observed": 3886000,
         "energy_modeled": 62286000, "energy_observed": 222117000,
         "ca_modeled": 49285000, "ca_observed": 126979000,
         "fe_modeled": 4317500, "fe_observed": 12983900,
         "mg_modeled": 77000000, "mg_observed": 231560000,
         "ph_modeled": 193600000, "ph_observed": 582208000,
         "k_modeled": 196465000, "k_observed": 732079000,
         "na_modeled": 550000, "na_observed": 1654000,
         "zn_modeled": 1347500, "zn_observed": 4052300,
         "cu_modeled": 467900, "cu_observed": 1434800,
         "fl_modeled": 1290000, "fl_observed": 4341000,
         "mn_modeled": 1216100, "mn_observed": 3444800,
         "se_modeled": 63900, "se_observed": 173700,
         "vita_modeled": 825000, "vita_observed": 2481000,
         "betac_modeled": 4400000, "betac_observed": 13232000,
         "alphac_modeled": 395900, "alphac_observed": 1310600,
         "vite_modeled": 220000, "vite_observed": 661600,
         "crypto_modeled": 258000, "crypto_observed": 868200,
         "lycopene_modeled": 88080, "lycopene_observed": 270420,
         "lutein_modeled": 16961000, "lutein_observed": 51191000,
         "betat_modeled": 137500, "betat_observed": 413500,
         "gammat_modeled": 613900, "gammat_observed": 1827700,
         "deltat_modeled": 395100, "deltat_observed": 1252800,
         "vitc_modeled": 1178400, "vitc_observed": 3894600,
         "thiamin_modeled": 113640, "thiamin_observed": 339900,
         "riboflavin_modeled": 316640, "riboflavin_observed": 1042700,
         "niacin_modeled": 2983000, "niacin_observed": 8601400,
         "pantothenic_modeled": 251140, "pantothenic_observed": 753400,
         "vitb6_modeled": 1113000, "vitb6_observed": 2977800,
         "folate_modeled": 91315000, "folate_observed": 281995000,
         "vitb12_modeled": 732000, "vitb12_observed": 2109000,
         "vitk_modeled": 11457000, "vitk_observed": 34362000},
        {"FID": 1, "corn_modeled": 40, "corn_observed": 80,
         "soybean_modeled": 70, "soybean_observed": 120,
         "protein_modeled": 36645000, "protein_observed": 67284000,
         "lipid_modeled": 4250000, "lipid_observed": 8136000,
         "energy_modeled": 240135000, "energy_observed": 462252000,
         "ca_modeled": 173753000, "ca_observed": 300732000,
         "fe_modeled": 15841300, "fe_observed": 28825200,
         "mg_modeled": 282520000, "mg_observed": 514080000,
         "ph_modeled": 710336000, "ph_observed": 1292544000,
         "k_modeled": 767933000, "k_observed": 1500012000,
         "na_modeled": 2018000, "na_observed": 3672000,
         "zn_modeled": 4944100, "zn_observed": 8996400,
         "cu_modeled": 1726000, "cu_observed": 3160800,
         "fl_modeled": 4887000, "fl_observed": 9228000,
         "mn_modeled": 4391200, "mn_observed": 7836000,
         "se_modeled": 228300, "se_observed": 402000,
         "vita_modeled": 3027000, "vita_observed": 5508000,
         "betac_modeled": 16144000, "betac_observed": 29376000,
         "alphac_modeled": 1492600, "alphac_observed": 2803200,
         "vite_modeled": 807200, "vite_observed": 1468800,
         "crypto_modeled": 977400, "crypto_observed": 1845600,
         "lycopene_modeled": 325020, "lycopene_observed": 595440,
         "lutein_modeled": 62293000, "lutein_observed": 113484000,
         "betat_modeled": 504500, "betat_observed": 918000,
         "gammat_modeled": 2246300, "gammat_observed": 4074000,
         "deltat_modeled": 1471200, "deltat_observed": 2724000,
         "vitc_modeled": 4440600, "vitc_observed": 8335200,
         "thiamin_modeled": 416340, "thiamin_observed": 756240,
         "riboflavin_modeled": 1191940, "riboflavin_observed": 2234640,
         "niacin_modeled": 10821800, "niacin_observed": 19423200,
         "pantothenic_modeled": 920840, "pantothenic_observed": 1674240,
         "vitb6_modeled": 3960600, "vitb6_observed": 6938400,
         "folate_modeled": 337505000, "folate_observed": 619500000,
         "vitb12_modeled": 2655000, "vitb12_observed": 4764000,
         "vitk_modeled": 42006000, "vitk_observed": 76368000}
    ], dtype=float)


def aggregate_pctl_polygons_table():
    """Generate expected aggregated nutrient results by polygon (e.g., FID).

    This function returns the expected DataFrame output of the
    `aggregate_to_polygons` function for use in the
    `test_aggregate_to_polygons` unit test. The DataFrame contains observed
    and modeled (at different percentiles) crop yields and nutrient totals,
    aggregated by polygon (FID).

    Returns:
        pandas.DataFrame: A table with observed and percentile-based modeled
            values for crop yield and nutrient totals, grouped by FID.

    """

    data = [
        [0, 0.25, 0.5, 0.75, 1, 0.5, 1, 1.5, 2, 247800, 495600, 743400,
         991200, 27700, 55400, 83100, 110800, 1557150, 3114300, 4671450,
         6228600, 1232125, 2464250, 3696375, 4928500, 107937.5, 215875,
         323812.5, 431750, 1925000, 3850000, 5775000, 7700000, 4840000,
         9680000, 14520000, 19360000, 4911625, 9823250, 14734875,
         19646500, 13750, 27500, 41250, 55000, 33687.5, 67375,
         101062.5, 134750, 11697.5, 23395, 35092.5, 46790, 32250,
         64500, 96750, 129000, 30402.5, 60805, 91207.5, 121610, 1597.5,
         3195, 4792.5, 6390, 20625, 41250, 61875, 82500, 110000, 220000,
         330000, 440000, 9897.5, 19795, 29692.5, 39590, 5500, 11000,
         16500, 22000, 6450, 12900, 19350, 25800, 2202, 4404, 6606,
         8808, 424025, 848050, 1272075, 1696100, 3437.5, 6875, 10312.5,
         13750, 15347.5, 30695, 46042.5, 61390, 9877.5, 19755, 29632.5,
         39510, 29460, 58920, 88380, 117840, 2841, 5682, 8523, 11364,
         7916, 15832, 23748, 31664, 74575, 149150, 223725, 298300,
         6278.5, 12557, 18835.5, 25114, 27825, 55650, 83475, 111300,
         2282875, 4565750, 6848625, 9131500, 18300, 36600, 54900,
         73200, 286425, 572850, 859275, 1145700],
        [1, 1.25, 2.5, 3.75, 4, 2.25, 4.5, 6.75, 7, 1163925, 2327850,
         3491775, 3664500, 133950, 267900, 401850, 425000, 7560525,
         15121050, 22681575, 24013500, 5575950, 11151900, 16727850,
         17375300, 503970, 1007940, 1511910, 1584130, 8988000,
         17976000, 26964000, 28252000, 22598400, 45196800, 67795200,
         71033600, 24109950, 48219900, 72329850, 76793300, 64200,
         128400, 192600, 201800, 157290, 314580, 471870, 494410,
         54847.5, 109695, 164542.5, 172600, 154425, 308850, 463275,
         488700, 140182.5, 280365, 420547.5, 439120, 7305, 14610,
         21915, 22830, 96300, 192600, 288900, 302700, 513600, 1027200,
         1540800, 1614400, 47212.5, 94425, 141637.5, 149260, 25680,
         51360, 77040, 80720, 30885, 61770, 92655, 97740, 10327.5,
         20655, 30982.5, 32502, 1981350, 3962700, 5944050, 6229300,
         16050, 32100, 48150, 50450, 71505, 143010, 214515, 224630,
         46657.5, 93315, 139972.5, 147120, 140475, 280950, 421425,
         444060, 13249.5, 26499, 39748.5, 41634, 37714.5, 75429,
         113143.5, 119194, 345120, 690240, 1035360, 1082180, 29299.5,
         58599, 87898.5, 92084, 126840, 253680, 380520, 396060,
         10720500, 21441000, 32161500, 33750500, 84675, 169350, 254025,
         265500, 1336575, 2673150, 4009725, 4200600]
    ]

    columns = [
        "FID", "corn_25", "corn_50", "corn_75", "corn_observed",
        "soybean_25", "soybean_50", "soybean_75", "soybean_observed",
        "protein_25", "protein_50", "protein_75", "protein_observed",
        "lipid_25", "lipid_50", "lipid_75", "lipid_observed",
        "energy_25", "energy_50", "energy_75", "energy_observed",
        "ca_25", "ca_50", "ca_75", "ca_observed", "fe_25", "fe_50",
        "fe_75", "fe_observed", "mg_25", "mg_50", "mg_75",
        "mg_observed", "ph_25", "ph_50", "ph_75", "ph_observed",
        "k_25", "k_50", "k_75", "k_observed", "na_25", "na_50",
        "na_75", "na_observed", "zn_25", "zn_50", "zn_75",
        "zn_observed", "cu_25", "cu_50", "cu_75", "cu_observed",
        "fl_25", "fl_50", "fl_75", "fl_observed", "mn_25", "mn_50",
        "mn_75", "mn_observed", "se_25", "se_50", "se_75",
        "se_observed", "vita_25", "vita_50", "vita_75",
        "vita_observed", "betac_25", "betac_50", "betac_75",
        "betac_observed", "alphac_25", "alphac_50", "alphac_75",
        "alphac_observed", "vite_25", "vite_50", "vite_75",
        "vite_observed", "crypto_25", "crypto_50", "crypto_75",
        "crypto_observed", "lycopene_25", "lycopene_50", "lycopene_75",
        "lycopene_observed", "lutein_25", "lutein_50", "lutein_75",
        "lutein_observed", "betat_25", "betat_50", "betat_75",
        "betat_observed", "gammat_25", "gammat_50", "gammat_75",
        "gammat_observed", "deltat_25", "deltat_50", "deltat_75",
        "deltat_observed", "vitc_25", "vitc_50", "vitc_75",
        "vitc_observed", "thiamin_25", "thiamin_50", "thiamin_75",
        "thiamin_observed", "riboflavin_25", "riboflavin_50",
        "riboflavin_75", "riboflavin_observed", "niacin_25",
        "niacin_50", "niacin_75", "niacin_observed", "pantothenic_25",
        "pantothenic_50", "pantothenic_75", "pantothenic_observed",
        "vitb6_25", "vitb6_50", "vitb6_75", "vitb6_observed",
        "folate_25", "folate_50", "folate_75", "folate_observed",
        "vitb12_25", "vitb12_50", "vitb12_75", "vitb12_observed",
        "vitk_25", "vitk_50", "vitk_75", "vitk_observed"
    ]

    return pandas.DataFrame(data, columns=columns, dtype=float)
