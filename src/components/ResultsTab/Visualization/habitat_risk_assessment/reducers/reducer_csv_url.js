const csvUrl = (state = null, action) => {
  switch (action.type) {
    case 'CSV_UPLOADED':
      return action.payload
    default:
      return state
    }
  }

export default csvUrl;
