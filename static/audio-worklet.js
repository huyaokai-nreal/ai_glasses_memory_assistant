class Pcm16CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = options?.processorOptions || {};
    this.targetRate = Number(processorOptions.targetSampleRate) || 16000;
    this.ratio = sampleRate / this.targetRate;
    this.source = [];
    this.position = 0;
    this.output = [];
    this.chunkSamples = Number(processorOptions.chunkSamples) || 4096;
    this.port.onmessage = (event) => {
      if (event.data?.type !== "flush") return;
      this.emitPcm(this.output.length);
      this.port.postMessage({
        type: "flushed",
        requestId: String(event.data.requestId || ""),
      });
    };
  }

  emitPcm(sampleCount) {
    if (sampleCount <= 0) return;
    const pcm = Int16Array.from(this.output.splice(0, sampleCount));
    this.port.postMessage({ type: "pcm", buffer: pcm.buffer }, [pcm.buffer]);
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input?.length) return true;
    for (let index = 0; index < input.length; index += 1) {
      this.source.push(input[index]);
    }
    while (this.position + 1 < this.source.length) {
      const leftIndex = Math.floor(this.position);
      const fraction = this.position - leftIndex;
      const sample = this.source[leftIndex] * (1 - fraction) + this.source[leftIndex + 1] * fraction;
      const clamped = Math.max(-1, Math.min(1, sample));
      this.output.push(clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767));
      this.position += this.ratio;
      if (this.output.length >= this.chunkSamples) {
        this.emitPcm(this.chunkSamples);
      }
    }
    const consumed = Math.floor(this.position);
    if (consumed > 0) {
      this.source.splice(0, consumed);
      this.position -= consumed;
    }
    return true;
  }
}

registerProcessor("pcm16-capture", Pcm16CaptureProcessor);
