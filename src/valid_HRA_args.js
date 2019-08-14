const SAMPLE_DATA = 'C:/Users/dmf/projects/invest/data/invest-sample-data/HabitatRiskAssess/Input/'

const MODEL_NAME = 'habitat_risk_assessment'

const MODEL_ARGS = {
  aoi_vector_path: {
    argname: 'aoi_vector_path',
    value: SAMPLE_DATA + 'subregions.shp',
    type: 'text',
    valid: false,
    validationRules: {'required': true, 'rule': 'filepath'},
  },
  criteria_table_path: {
    argname: 'criteria_table_path',
    value: SAMPLE_DATA + 'exposure_consequence_criteria.csv',
    type: 'text',
    valid: false,
    validationRules: {'required': true, 'rule': 'filepath'},
  },
  decay_eq: {
    argname: 'decay_eq',
    value: 'Linear',
    type: 'select',
    options: ['Linear', 'Multiplicative'],
    valid: true,
    validationRules: {'required': true, 'rule': 'select'},
  },
  info_table_path: {
    argname: 'info_table_path',
    value: SAMPLE_DATA + 'habitat_stressor_info.csv',
    type: 'text',
    valid: false,
    validationRules: {'required': true, 'rule': 'filepath'},
  },
  max_rating: {
    argname: 'max_rating',
    value: '3',
    type: 'text',
    valid: false,
    validationRules: {'required': true, 'rule': 'integer'},
  },
  resolution: {
    argname: 'resolution',
    value: '500',
    type: 'text',
    valid: false,
    validationRules: {'required': true, 'rule': 'integer'},
  },
  risk_eq: {
    argname: 'risk_eq',
    value: 'Euclidean',
    type: 'select',
    options: ['Euclidean', 'Exponential'],
    valid: true,
    validationRules: {'required': true, 'rule': 'select'},
  },
  results_suffix: {
    argname: 'results_suffix',
    value: '',
    type: 'text',
    valid: true,
    validationRules: {'required': false, 'rule': 'string'},
  },
  workspace_dir: {
    argname: 'workspace_dir',
    value: 'workspace',
    type: 'text',
    valid: false,
    validationRules: {'required': true, 'rule': 'workspace'},
  },
  visualize_outputs: {
    argname: 'visualize_outputs',
    value: 'True',
    type: 'select',
    options: ['True', 'False'],
    valid: true,
    validationRules: {'required': true, 'rule': 'select'},
  },
};

export {MODEL_ARGS, MODEL_NAME};