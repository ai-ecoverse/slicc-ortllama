export class HttpError extends Error {
  constructor(status, message, { type, code = null, param = null } = {}) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.type =
      type ??
      (status === 401
        ? 'authentication_error'
        : status >= 500
          ? 'server_error'
          : 'invalid_request_error');
    this.code = code;
    this.param = param;
  }
}

export function errorBody(error) {
  if (error instanceof HttpError) {
    return {
      status: error.status,
      body: {
        error: { message: error.message, type: error.type, param: error.param, code: error.code },
      },
    };
  }
  const message = error instanceof Error ? error.message : String(error);
  return {
    status: 500,
    body: { error: { message, type: 'server_error', param: null, code: null } },
  };
}

export const notFound = (message, code = 'model_not_found') =>
  new HttpError(404, message, { code, param: 'model' });
export const badRequest = (message, param = null) => new HttpError(400, message, { param });
