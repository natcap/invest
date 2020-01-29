import { combineReducers } from 'redux';
import csvUrl from './reducer_csv_url';
import vectorsOnMap from './reducer_vectors_on_map';
import fileSuffix from './reducer_file_suffix';

const rootReducer = combineReducers({ csvUrl, vectorsOnMap, fileSuffix });

export default rootReducer;
