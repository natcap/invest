module.exports = {
  presets: [
    '@babel/preset-env',
    '@babel/preset-react', // needed only for tests
  ],
  plugins: [
    '@babel/plugin-transform-runtime',
  ],
};
