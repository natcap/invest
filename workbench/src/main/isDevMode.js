// defaultApp property is added by electron in devmode.
// It means the app is running in the "default" electron app
// as opposed to a purposely-built package, as in production.
export default !!process.defaultApp;
