const vectorsOnMap = (state = [], action) => {
  switch (action.type) {
    case 'VECTORS_UPDATED':
      return action.payload
    default:
      return state
    }
  }

export default vectorsOnMap;
