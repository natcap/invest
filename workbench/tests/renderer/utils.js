export function mockUISpec(spec, modelName) {
  return {
    [modelName]: { order: [Object.keys(spec.args)] },
  };
}
