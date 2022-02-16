class Store {
  constructor(name) {
    this.store = {
      name: name,
    };
  }

  getItem(key) {
    return new Promise((resolve) => resolve(this.store[key]));
  }

  setItem(key, val) {
    this.store[key] = val;
  }

  removeItem(key) {
    delete this.store[key];
  }

  clear() {
    const tmpName = this.store.name;
    this.store = {
      name: tmpName
    };
  }
}

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
  createInstance: function (driver) {
    return new Store(driver.name);
  },
};

export default localforage;
