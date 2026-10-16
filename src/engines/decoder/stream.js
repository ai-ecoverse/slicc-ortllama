const HOLD = 4;

export function toolCallId(random = () => crypto.randomUUID()) {
  return `call_${random().replaceAll('-', '').slice(0, 24)}`;
}

export function createParser({ decode, markers, special, emit, newId = toolCallId }) {
  let state = 'content';
  let pending = [];
  let name = '';
  let calls = 0;

  const route = (text) => {
    if (text === '') return;
    if (state === 'content') emit({ type: 'content', text });
    else if (state === 'think') emit({ type: 'reasoning', text });
    else if (state === 'name') name += text;
    else emit({ type: 'args', index: calls - 1, text });
  };

  const flush = () => {
    if (pending.length === 0) return;
    const text = decode(pending);
    pending = [];
    route(text);
  };

  const startCall = () => {
    emit({ type: 'tool', index: calls++, id: newId(), name: name.trim() });
    name = '';
    state = 'args';
  };

  return {
    push(id) {
      if (id === markers.toolCall) {
        flush();
        if (state === 'name') {
          startCall();
          emit({ type: 'args', index: calls - 1, text: '{}' });
        }
        state = 'name';
        name = '';
        return;
      }
      if (id === markers.args && state === 'name') {
        flush();
        startCall();
        return;
      }
      if (id === markers.think && state === 'content') {
        flush();
        state = 'think';
        return;
      }
      if (id === markers.thinkEnd && state === 'think') {
        flush();
        state = 'content';
        return;
      }
      if (special.has(id)) return;
      pending.push(id);
      const text = decode(pending);
      if (text.endsWith('�') && pending.length < HOLD) return;
      pending = [];
      route(text);
    },
    end() {
      flush();
      if (state === 'name') {
        startCall();
        emit({ type: 'args', index: calls - 1, text: '{}' });
      }
      return { toolCalls: calls };
    },
    get state() {
      return state;
    },
  };
}

export function stopFilter(stops, emit) {
  const list = stops.filter((s) => s.length > 0);
  let buffer = '';
  let stopped = false;
  const longest = Math.max(0, ...list.map((s) => s.length));
  return {
    push(text) {
      if (stopped) return true;
      if (list.length === 0) {
        emit(text);
        return false;
      }
      buffer += text;
      let at = -1;
      for (const s of list) {
        const i = buffer.indexOf(s);
        if (i >= 0 && (at < 0 || i < at)) at = i;
      }
      if (at >= 0) {
        if (at > 0) emit(buffer.slice(0, at));
        buffer = '';
        stopped = true;
        return true;
      }
      let keep = 0;
      for (let n = Math.min(longest - 1, buffer.length); n > 0; n--) {
        const tail = buffer.slice(-n);
        if (list.some((s) => s.startsWith(tail))) {
          keep = n;
          break;
        }
      }
      const out = buffer.slice(0, buffer.length - keep);
      buffer = buffer.slice(buffer.length - keep);
      if (out) emit(out);
      return false;
    },
    end() {
      if (!stopped && buffer) emit(buffer);
      buffer = '';
    },
  };
}
