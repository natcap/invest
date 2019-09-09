const fileSuffix = (state = '', action) => {
  switch (action.type) {
    case 'SUFFIX_OBTAINED':
      return action.payload
    default:
      return state
    }
  }

export default fileSuffix;
