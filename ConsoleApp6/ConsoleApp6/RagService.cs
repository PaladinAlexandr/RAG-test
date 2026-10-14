using System.Text;
using System.Text.Json;

namespace ConsoleApp6;

public sealed class RagService
{
    private const int MaxDocumentBytes = 2 * 1024 * 1024;
    private readonly RagOptions options;
    private readonly OllamaClient ollama;
    private readonly SemaphoreSlim writeLock = new(1, 1);
    private readonly string knowledgeDirectory;
    private readonly string indexPath;
    private KnowledgeIndex index;

    public RagService(RagOptions options, OllamaClient ollama, IHostEnvironment environment)
    {
        this.options = options;
        this.ollama = ollama;
        knowledgeDirectory = Path.GetFullPath(options.KnowledgeDirectory, environment.ContentRootPath);
        indexPath = Path.GetFullPath(options.IndexPath, environment.ContentRootPath);
        Directory.CreateDirectory(knowledgeDirectory);
        index = File.Exists(indexPath)
            ? JsonSerializer.Deserialize<KnowledgeIndex>(File.ReadAllText(indexPath))
              ?? throw new InvalidOperationException("Индекс пуст. Удалите файл индекса и выполните /api/reindex.")
            : NewIndex([]);
        if (index.Chunks is null || index.Chunks.Any(c => c is null || c.Vector is null || c.Vector.Length == 0 ||
            c.Vector.Any(x => !float.IsFinite(x)) || c.Vector.All(x => x == 0) ||
            string.IsNullOrWhiteSpace(c.Text) || string.IsNullOrWhiteSpace(c.Source)) ||
            index.Chunks.Select(c => c.Vector.Length).Distinct().Count() > 1)
            throw new InvalidOperationException("Индекс повреждён. Удалите файл индекса и выполните /api/reindex.");
    }

    public object Status()
    {
        var snapshot = Volatile.Read(ref index);
        return new
        {
            documents = snapshot.Chunks.Select(c => c.Source).Distinct().Order().ToArray(),
            chunks = snapshot.Chunks.Length,
            needsReindex = !Compatible(snapshot),
            knowledgeDirectory, indexPath, options.ChatModel, options.EmbeddingModel
        };
    }

    public async Task<object> ReindexAsync(CancellationToken ct)
    {
        await writeLock.WaitAsync(ct);
        try
        {
            var chunks = new List<KnowledgeChunk>();
            foreach (var path in Directory.EnumerateFiles(knowledgeDirectory, "*", SearchOption.AllDirectories)
                         .Where(IsSupported).Order(StringComparer.Ordinal))
            {
                ct.ThrowIfCancellationRequested();
                if (new FileInfo(path).Length > MaxDocumentBytes)
                    throw new ArgumentException($"Файл {Path.GetFileName(path)} больше 2 МБ.");
                var text = await File.ReadAllTextAsync(path, Encoding.UTF8, ct);
                var source = Path.GetRelativePath(knowledgeDirectory, path).Replace('\\', '/');
                chunks.AddRange(await EmbedDocumentAsync(source, text, ct));
            }
            await SaveAsync(NewIndex(chunks.ToArray()), ct);
            return Status();
        }
        finally { writeLock.Release(); }
    }

    // API documents live in the JSON index; folder documents are imported by reindex.
    public async Task<object> AddDocumentAsync(string name, string text, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(name) || name.Length > 200 ||
            name.IndexOfAny(['/', '\\']) >= 0 || name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || !IsSupported(name))
            throw new ArgumentException("Name: имя файла .txt или .md без пути, до 200 символов.");
        if (string.IsNullOrWhiteSpace(text) || Encoding.UTF8.GetByteCount(text) > MaxDocumentBytes)
            throw new ArgumentException("Text: непустой текст UTF-8 размером до 2 МБ.");
        await writeLock.WaitAsync(ct);
        try
        {
            var snapshot = Volatile.Read(ref index);
            EnsureCompatible(snapshot);
            var chunks = await EmbedDocumentAsync(name, text, ct);
            var updated = snapshot.Chunks.Where(c => !c.Source.Equals(name, StringComparison.OrdinalIgnoreCase))
                .Concat(chunks).ToArray();
            await SaveAsync(NewIndex(updated), ct);
            return new { document = name, chunks = chunks.Length, totalChunks = updated.Length };
        }
        finally { writeLock.Release(); }
    }

    public async Task<AskResult> AskAsync(string question, int? topK, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(question) || question.Length > 2000)
            throw new ArgumentException("Question: непустой вопрос до 2000 символов.");
        var count = topK ?? options.TopK;
        if (count < 1 || count > 8) throw new ArgumentException("TopK должен быть от 1 до 8.");
        var snapshot = Volatile.Read(ref index);
        EnsureCompatible(snapshot);
        if (snapshot.Chunks.Length == 0)
            return new("База знаний пустая. Добавьте документы или выполните /api/reindex.", []);
        var queryText = options.EmbeddingModel.StartsWith("qwen3-embedding", StringComparison.OrdinalIgnoreCase)
            ? $"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {question}"
            : question;
        var query = (await ollama.EmbedAsync([queryText], ct))[0];
        if (snapshot.Chunks[0].Vector.Length != query.Length)
            throw new InvalidOperationException("Размер embeddings изменился. Выполните /api/reindex.");
        var sources = snapshot.Chunks.Select(c => new { Chunk = c, Score = Cosine(c.Vector, query) })
            .Where(x => x.Score >= options.MinSimilarity).OrderByDescending(x => x.Score).Take(count)
            .Select((x, i) => new Source(i + 1, x.Chunk.Source, x.Chunk.Number, x.Score, x.Chunk.Text)).ToArray();
        if (sources.Length == 0)
            return new("В базе знаний не найдено подходящих фрагментов для ответа.", []);
        var context = string.Join("\n\n", sources.Select(s => $"[{s.Id}] {s.Document}, фрагмент {s.Chunk}\n{s.Text}"));
        return new(await ollama.AnswerAsync(question, context, ct), sources);
    }

    public static string[] SplitText(string text, int size, int overlap)
    {
        if (size < 2 || overlap < 0 || overlap >= size) throw new ArgumentException("Некорректный размер фрагментов.");
        text = text.Replace("\r\n", "\n").Replace('\r', '\n').Trim();
        var chunks = new List<string>();
        for (var start = 0; start < text.Length;)
        {
            var end = Math.Min(start + size, text.Length);
            if (end < text.Length)
            {
                var boundary = end;
                while (boundary > start + Math.Max(size / 2, overlap + 1) && !char.IsWhiteSpace(text[boundary - 1])) boundary--;
                if (char.IsWhiteSpace(text[boundary - 1])) end = boundary;
                if (end < text.Length && char.IsHighSurrogate(text[end - 1]) && char.IsLowSurrogate(text[end])) end--;
            }
            var chunk = text[start..end].Trim();
            if (chunk.Length > 0) chunks.Add(chunk);
            if (end == text.Length) break;
            start = Math.Max(start + 1, end - overlap);
            if (start < text.Length && char.IsLowSurrogate(text[start]) && start > 0 && char.IsHighSurrogate(text[start - 1])) start++;
        }
        return chunks.ToArray();
    }

    private async Task<KnowledgeChunk[]> EmbedDocumentAsync(string source, string text, CancellationToken ct)
    {
        var parts = SplitText(text, options.ChunkSize, options.ChunkOverlap);
        var chunks = new List<KnowledgeChunk>();
        for (var offset = 0; offset < parts.Length; offset += 16)
        {
            var batch = parts.Skip(offset).Take(16).ToArray();
            var vectors = await ollama.EmbedAsync(batch, ct);
            chunks.AddRange(batch.Select((part, i) => new KnowledgeChunk(source, offset + i + 1, part, vectors[i])));
        }
        return chunks.ToArray();
    }

    private async Task SaveAsync(KnowledgeIndex updated, CancellationToken ct)
    {
        if (updated.Chunks.Select(c => c.Vector.Length).Distinct().Count() > 1)
            throw new InvalidOperationException("Размер embeddings изменился. Выполните /api/reindex.");
        Directory.CreateDirectory(Path.GetDirectoryName(indexPath)!);
        var temporary = indexPath + ".tmp";
        try
        {
            await using (var stream = File.Create(temporary))
                await JsonSerializer.SerializeAsync(stream, updated, cancellationToken: ct);
            ct.ThrowIfCancellationRequested();
            File.Move(temporary, indexPath, overwrite: true);
            Volatile.Write(ref index, updated);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    private KnowledgeIndex NewIndex(KnowledgeChunk[] chunks) =>
        new(1, options.EmbeddingModel, options.ChunkSize, options.ChunkOverlap, chunks);
    private bool Compatible(KnowledgeIndex value) => value.Version == 1 && value.EmbeddingModel == options.EmbeddingModel &&
        value.ChunkSize == options.ChunkSize && value.ChunkOverlap == options.ChunkOverlap;
    private void EnsureCompatible(KnowledgeIndex value)
    {
        if (!Compatible(value)) throw new InvalidOperationException("Настройки индекса изменились. Выполните /api/reindex.");
    }
    private static bool IsSupported(string path) => Path.GetExtension(path).Equals(".txt", StringComparison.OrdinalIgnoreCase) ||
        Path.GetExtension(path).Equals(".md", StringComparison.OrdinalIgnoreCase);
    private static double Cosine(float[] left, float[] right)
    {
        double dot = 0, a = 0, b = 0;
        for (var i = 0; i < left.Length; i++)
        {
            dot += (double)left[i] * right[i];
            a += (double)left[i] * left[i];
            b += (double)right[i] * right[i];
        }
        return dot / Math.Sqrt(a * b);
    }
}

public record KnowledgeIndex(int Version, string EmbeddingModel, int ChunkSize, int ChunkOverlap, KnowledgeChunk[] Chunks);
public record KnowledgeChunk(string Source, int Number, string Text, float[] Vector);
public record Source(int Id, string Document, int Chunk, double Score, string Text);
public record AskResult(string Answer, Source[] Sources);
