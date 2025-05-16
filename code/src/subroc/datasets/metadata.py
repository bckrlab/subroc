from enum import Enum
import numpy as np

from subroc.datasets.loader import DatasetLoader, UCIDatasetLoader, OpenMLDatasetLoader

class DatasetName(str, Enum):
    SCaPE_ADULT = "Adult"
    SCaPE_CREDIT_A = "Credit Approval"
    SCaPE_HABERMAN = "Haberman's Survival"
    SCaPE_IONOSPHERE = "Ionosphere"
    SCaPE_LABOR = "Labor Relations"
    SCaPE_MUSHROOM = "Mushroom"
    SCaPE_TIC_TAC_TOE = "Tic-Tac-Toe Endgame"
    SCaPE_WISCONSIN = "Breast Cancer Wisconsin"
    CREDIT_G = "Statlog (German Credit Data)"
    SYNTH_XOR = "Synthetic XOR"
    OpenML_ADULT = "OpenML Adult"
    UCI_MUSHROOM = "UCI Mushroom"
    UCI_WISCONSIN = "UCI Breast Cancer Wisconsin"
    UCI_CENSUS_KDD = "UCI Census-Income (KDD)"
    UCI_CREDIT_A = "UCI Credit Approval"
    UCI_BANK_MARKETING = "UCI Bank Marketing"
    UCI_CREDIT_CARD_CLIENTS = "UCI Credit Card Clients"
    UCI_DIABETES = "UCI Diabetes 130-US Hospitals 1999-2008"  # issue: has at least one missing value in every row


scape_dataset_names = [
    DatasetName.SCaPE_ADULT,
    DatasetName.SCaPE_CREDIT_A,
    DatasetName.SCaPE_HABERMAN,
    DatasetName.SCaPE_IONOSPHERE,
    DatasetName.SCaPE_LABOR,
    DatasetName.SCaPE_MUSHROOM,
    DatasetName.SCaPE_TIC_TAC_TOE,
    DatasetName.SCaPE_WISCONSIN
]


def to_DatasetName(dataset_name):
    for name_enum in DatasetName:
        if str.lower(dataset_name) == str.lower(name_enum):
            return name_enum
    
    return None


class DatasetMetadata:
    def __init__(self, name, dataset_dir, gt_name, gt_true_value, score_name, number_of_columns, raw_filename,
                 categorical_attributes: list[str] = None, missing_value_symbol: str | None = "?", has_test_split=False,
                 skip_rows_raw_test=0, skip_rows_raw_train=0,
                 skip_rows_raw=0, skip_rows_processed=None,
                 forced_nominal_attributes_raw_test=None, forced_nominal_attributes_raw_train=None,
                 forced_nominal_attributes_raw=None, forced_nominal_attributes_processed=None,
                 forced_numerical_attributes_raw_test=None, forced_numerical_attributes_raw_train=None,
                 forced_numerical_attributes_raw=None, forced_numerical_attributes_processed=None,
                 replace_values_raw_test=None, replace_values_raw_train=None, replace_values_raw=None,
                 replace_values_processed=None, id_attribute=None, read_raw_column_names=False,
                 attributes_to_drop: list[str] | None = None, loader: DatasetLoader = None):
        self.name = name
        self.dataset_dir = dataset_dir
        self.gt_name = gt_name
        self.gt_true_value = gt_true_value
        self.score_name = score_name
        self.number_of_columns = number_of_columns
        self.raw_filename = raw_filename
        self.categorical_attributes = categorical_attributes
        self.missing_value_symbol = missing_value_symbol
        self.has_test_split = has_test_split

        if skip_rows_raw:
            self.skip_rows_raw_test = skip_rows_raw
            self.skip_rows_raw_train = skip_rows_raw
        else:
            self.skip_rows_raw_test = skip_rows_raw_test
            self.skip_rows_raw_train = skip_rows_raw_train
        self.skip_rows_processed = skip_rows_processed

        if forced_nominal_attributes_raw is not None:
            self.forced_nominal_attributes_raw_test = forced_nominal_attributes_raw
            self.forced_nominal_attributes_raw_train = forced_nominal_attributes_raw
        else:
            self.forced_nominal_attributes_raw_test = forced_nominal_attributes_raw_test
            self.forced_nominal_attributes_raw_train = forced_nominal_attributes_raw_train
        self.forced_nominal_attributes_processed = forced_nominal_attributes_processed

        if forced_numerical_attributes_raw is not None:
            self.forced_numerical_attributes_raw_test = forced_numerical_attributes_raw
            self.forced_numerical_attributes_raw_train = forced_numerical_attributes_raw
        else:
            self.forced_numerical_attributes_raw_test = forced_numerical_attributes_raw_test
            self.forced_numerical_attributes_raw_train = forced_numerical_attributes_raw_train
        self.forced_numerical_attributes_processed = forced_numerical_attributes_processed

        if replace_values_raw is not None:
            self.replace_values_raw_test = replace_values_raw
            self.replace_values_raw_train = replace_values_raw
        else:
            self.replace_values_raw_test = replace_values_raw_test
            self.replace_values_raw_train = replace_values_raw_train
        self.replace_values_processed = replace_values_processed

        self.id_attribute = id_attribute

        self.prediction_type = None

        self.read_raw_column_names = read_raw_column_names

        self.attributes_to_drop = attributes_to_drop

        self.loader = loader

    def __str__(self):
        return self.name


meta_dict = {
    DatasetName.SCaPE_ADULT: DatasetMetadata(
        DatasetName.SCaPE_ADULT,
        "adult/",
        "att15",
        "<=50K",
        "confidence(<=50K)",
        15,
        "adult",
        has_test_split=True,
        skip_rows_raw_test=1,
        replace_values_raw_test={"att15": {"<=50K.": "<=50K", ">50K.": ">50K"}}),

    DatasetName.SCaPE_CREDIT_A: DatasetMetadata(
        DatasetName.SCaPE_CREDIT_A,
        "credit_approval/",
        "att16",
        "+",
        "confidence(+)",
        16,
        "crx",
        forced_numerical_attributes_raw=["att2", "att14"]),

    DatasetName.SCaPE_HABERMAN: DatasetMetadata(
        DatasetName.SCaPE_HABERMAN,
        "haberman_s_survival/",
        "att4",
        1,
        "confidence(true)",
        4,
        "haberman",
        forced_nominal_attributes_processed=["att2"]),

    DatasetName.SCaPE_IONOSPHERE: DatasetMetadata(
        DatasetName.SCaPE_IONOSPHERE,
        "ionosphere/",
        "att35",
        "g",
        "confidence(g)",
        35,
        "ionosphere"),

    DatasetName.SCaPE_LABOR: DatasetMetadata(
        DatasetName.SCaPE_LABOR,
        "labor_relations/",
        "att17",
        "good",
        "confidence(good)",
        17,
        "C4.5/labor-neg",
        has_test_split=True,
        forced_nominal_attributes_raw_test=["att1", "att2"],
        forced_numerical_attributes_raw=["att1", "att2", "att3", "att4", "att6", "att8", "att9", "att11"]),

    DatasetName.SCaPE_MUSHROOM: DatasetMetadata(
        DatasetName.SCaPE_MUSHROOM,
        "mushroom/",
        "att1",
        "p",
        "confidence(p)",
        23,
        "agaricus-lepiota"),

    DatasetName.SCaPE_TIC_TAC_TOE: DatasetMetadata(
        DatasetName.SCaPE_TIC_TAC_TOE,
        "tic_tac_toe_endgame/",
        "att10",
        "positive",
        "confidence(positive)",
        10,
        "tic-tac-toe"),

    DatasetName.SCaPE_WISCONSIN: DatasetMetadata(
        DatasetName.SCaPE_WISCONSIN,
        "breast_cancer_wisconsin_original/",
        "att11",
        4,
        "confidence(true)",
        11,
        "breast-cancer-wisconsin",
        forced_numerical_attributes_raw=["att7"],
        id_attribute="att1"),

    DatasetName.CREDIT_G: DatasetMetadata(
        DatasetName.CREDIT_G,
        "credit_g/",
        "class",
        1,
        "score",
        21,
        "credit-g",
        categorical_attributes=[f"Attribute{att_num}" for att_num in [1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20]],
        missing_value_symbol=None,
        read_raw_column_names=True,
        loader=UCIDatasetLoader(144)),

    DatasetName.SYNTH_XOR: DatasetMetadata(
        DatasetName.SYNTH_XOR,
        "synth_xor/",
        "class",
        1,
        "score",
        3,
        "synth_xor",
        missing_value_symbol=None,
        read_raw_column_names=True),

    DatasetName.OpenML_ADULT: DatasetMetadata(
        DatasetName.OpenML_ADULT,
        "openml_adult/",
        "class",
        "<=50K",
        "score",
        15,
        "adult",
        categorical_attributes=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                'sex', 'native-country'],
        read_raw_column_names=True,
        loader=OpenMLDatasetLoader(1590)),
    
    DatasetName.UCI_MUSHROOM: DatasetMetadata(
        DatasetName.UCI_MUSHROOM,
        "uci_mushroom/",
        "poisonous",
        "p",
        "confidence(p)",
        None,
        "agaricus-lepiota",
        categorical_attributes=["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
                                "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                                "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                                "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"],
        loader=UCIDatasetLoader(73)),

    DatasetName.UCI_WISCONSIN: DatasetMetadata(
        DatasetName.UCI_WISCONSIN,
        "uci_breast_cancer_wisconsin_original/",
        "Class",
        4,
        "confidence(true)",
        None,
        "breast-cancer-wisconsin",
        attributes_to_drop=["Sample_code_number"],
        loader=UCIDatasetLoader(15)),

    DatasetName.UCI_CENSUS_KDD: DatasetMetadata(
        DatasetName.UCI_CENSUS_KDD,
        "uci_census_kdd/",
        "income",
        "-50000",
        "confidence(true)",
        None,
        "census_kdd",
        categorical_attributes=["ACLSWKR", "AHGA", "AHSCOL", "AMARITL", "AMJIND", "AMJOCC", "ARACE", "AREORGN", "ASEX", "AUNMEM", "AUNTYPE",
                                "AWKSTAT", "FILESTAT", "GRINREG", "GRINST", "HHDFMX", "HHDREL", "MIGMTR1", "MIGMTR3", "MIGMTR4", "MIGSAME", "MIGSUN", "NOEMP",
                                "PARENT", "PEFNTVTY", "PEMNTVTY", "PENATVTY", "PRCITSHP", "VETQVA", "VETYN"],
        attributes_to_drop=["MARSUPWRT"],
        loader=UCIDatasetLoader(117)),

    DatasetName.UCI_CREDIT_A: DatasetMetadata(
        DatasetName.UCI_CREDIT_A,
        "credit_approval/",
        "A16",
        "+",
        "confidence(+)",
        None,
        "credit_approval",
        categorical_attributes=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"],
        loader=UCIDatasetLoader(27)),

    DatasetName.UCI_BANK_MARKETING: DatasetMetadata(
        DatasetName.UCI_BANK_MARKETING,
        "bank_marketing/",
        "y",
        "yes",
        "confidence(yes)",
        None,
        "bank_marketing",
        categorical_attributes=["job", "marital", "education", "default", "housing", "loan", "contact", "day_of_week", "month", "poutcome"],
        loader=UCIDatasetLoader(222)),

    DatasetName.UCI_CREDIT_CARD_CLIENTS: DatasetMetadata(
        DatasetName.UCI_CREDIT_CARD_CLIENTS,
        "credit_card_clients/",
        "Y",
        1,
        "confidence(1)",
        None,
        "credit_card_clients",
        attributes_to_drop=["ID"],
        loader=UCIDatasetLoader(350)),

    DatasetName.UCI_DIABETES: DatasetMetadata(
        DatasetName.UCI_DIABETES,
        "diabetes/",
        "readmitted",
        "<30",
        "confidence(<30)",
        None,
        "diabetes",
        categorical_attributes=["race", "gender", "age", "weight", "admission_type_id", "discharge_disposition_id", "admission_source_id",
                                "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult", "metformin",
                                "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
                                "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
                                "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
                                "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed"],
        attributes_to_drop=["encounter_id", "patient_nbr"],
        loader=UCIDatasetLoader(296)),
}
