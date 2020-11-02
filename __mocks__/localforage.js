const localforage = {
  store: {},
  getItem: function (key) {
    return new Promise((resolve) => resolve(localforage.store[key]));
  },
  setItem: function (key, val) {
    localforage.store[key] = val;
  },
  clear: function() {
    localforage.store = {};
  },
};

export default localforage;
