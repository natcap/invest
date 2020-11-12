const localforage = {
  store: {},
  getItem: function (key) {
    return new Promise((resolve) => resolve(localforage.store[key]));
  },
  setItem: function (key, val) {
    localforage.store[key] = val;
  },
  removeItem: function (key) {
    delete localforage.store[key];
  },
  clear: function () {
    localforage.store = {};
  },
};

export default localforage;
