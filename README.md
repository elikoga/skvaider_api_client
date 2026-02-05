Auffällt:
prod ist embeddings1
dev ist embeddings2

Embeddings2 liefert mehr Nachkommastellen und das Ergebnis größer
188K und 319K

Ich habe 6 Datenpunkte getestet.

Einfach mal mit auf das System, und später dann in fc-nixos? Oder eher in Skvaider

- einfach ins fc-nixos, ike00

---

Verschiedene Modi:
- streaming
- chat completions vs completions api
- sustained rate vs N batches of requests
- embeddings - batched one request vs many requests
- input format: Ordner mit JSONs (request bodys)
- input dataset: can try https://huggingface.co/datasets/google-research-datasets/mbpp

Metriken zu messen:
- time to first token (streaming)
- output tokens per second
- input tokens per second
