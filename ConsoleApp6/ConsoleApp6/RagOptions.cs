namespace ConsoleApp6;

public sealed class RagOptions
{
    public string OllamaUrl { get; set; } = "http://localhost:11434";
    public string ChatModel { get; set; } = "qwen3:8b";
    public string EmbeddingModel { get; set; } = "qwen3-embedding:0.6b";
    public string KnowledgeDirectory { get; set; } = "knowledge";
    public string IndexPath { get; set; } = "data/index.json";
    public int ChunkSize { get; set; } = 1200;
    public int ChunkOverlap { get; set; } = 200;
    public int TopK { get; set; } = 4;
    public double MinSimilarity { get; set; } = 0.3;
    public int TimeoutSeconds { get; set; } = 300;

    public void Validate()
    {
        if (!Uri.TryCreate(OllamaUrl, UriKind.Absolute, out var uri) ||
            (uri.Scheme != "http" && uri.Scheme != "https"))
            throw new ArgumentException("Rag:OllamaUrl должен быть HTTP/HTTPS адресом.");
        if (string.IsNullOrWhiteSpace(ChatModel) || string.IsNullOrWhiteSpace(EmbeddingModel) ||
            string.IsNullOrWhiteSpace(KnowledgeDirectory) || string.IsNullOrWhiteSpace(IndexPath))
            throw new ArgumentException("Укажите модели и пути в разделе Rag.");
        if (ChunkSize < 200 || ChunkSize > 4000 || ChunkOverlap < 0 || ChunkOverlap >= ChunkSize)
            throw new ArgumentException("ChunkSize: 200–4000; ChunkOverlap: от 0 до ChunkSize - 1.");
        if (TopK < 1 || TopK > 8 || !double.IsFinite(MinSimilarity) || MinSimilarity < -1 || MinSimilarity > 1 ||
            TimeoutSeconds < 1 || TimeoutSeconds > 3600)
            throw new ArgumentException("Проверьте TopK (1–8), MinSimilarity (-1–1) и TimeoutSeconds (1–3600).");
    }
}
