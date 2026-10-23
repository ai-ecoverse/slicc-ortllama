import { registerHooks } from 'node:module';

registerHooks({
  resolve(specifier, context, next) {
    if (specifier === 'onnxruntime-node' || specifier.startsWith('onnxruntime-node/'))
      throw Object.assign(new Error(`Cannot find package '${specifier}'`), {
        code: 'ERR_MODULE_NOT_FOUND',
      });
    return next(specifier, context);
  },
});
