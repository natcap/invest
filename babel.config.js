module.exports = {
  presets: [
    ['@babel/preset-env', {
      targets: {
        electron: '12'
      }
    }],
    '@babel/preset-react',
  ],
  plugins: [
    '@babel/plugin-transform-runtime',
  ],
};
