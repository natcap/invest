export default class Store {
  constructor(options) {
    this.defaults = options.defaults || {};
    this.store = this.defaults;
  }

  get(key) {
    return this.store[key];
  }

  set(key, val) {
    this.store[key] = val;
  }

  delete(key) {
    delete this.store[key];
  }

  reset() {
    this.store = this.defaults;
  }
}
