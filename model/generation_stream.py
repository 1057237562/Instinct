"""Chunked token transfers and EOS checks for native autoregressive decoding."""
import torch


class TokenChunkBuffer:
    def __init__(self, streamer, chunk_size=16, eos_token_id=None):
        if chunk_size < 1:
            raise ValueError('stream_chunk_size must be >= 1')
        self.streamer = streamer
        self.chunk_size = chunk_size
        self.eos_token_id = eos_token_id
        self.pending = []
        self.overshoot = 0

    def push(self, token, finished, final=False):
        self.pending.append(token)
        if len(self.pending) < self.chunk_size and not final:
            return False
        # Tokens and completion flags travel together in one device->CPU copy.
        block = torch.cat([*self.pending, finished[:, None].to(token.dtype)], dim=1).cpu()
        self.pending.clear()
        tokens, flags = block[:, :-1], block[:, -1]
        done = bool(flags.bool().all()) if self.eos_token_id is not None else False
        if done:
            # Finished rows emit only EOS. Trim the extra masked steps executed
            # since the final sequence actually finished, preserving API output.
            positions = tokens.eq(self.eos_token_id).to(torch.int64).argmax(dim=1)
            keep = int(positions.max()) + 1
            self.overshoot = tokens.shape[1] - keep
            tokens = tokens[:, :keep]
        if self.streamer is not None:
            self.streamer.put(tokens)
        return done
