using ConsoleApp6;

var builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(server => server.Limits.MaxRequestBodySize = 3 * 1024 * 1024);
var options = builder.Configuration.GetSection("Rag").Get<RagOptions>() ?? new();
options.Validate();
builder.Services.AddSingleton(options);
builder.Services.AddHttpClient<OllamaClient>(client =>
{
    client.BaseAddress = new Uri(options.OllamaUrl.TrimEnd('/') + "/");
    client.Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds);
});
builder.Services.AddSingleton<RagService>();
var app = builder.Build();
_ = app.Services.GetRequiredService<RagService>();
app.Use(async (context, next) =>
{
    try { await next(context); }
    catch (Exception ex) when (ex is ArgumentException or InvalidOperationException
                              or HttpRequestException or OperationCanceledException)
    {
        if (context.RequestAborted.IsCancellationRequested) return;
        var status = ex switch
        {
            ArgumentException => 400,
            InvalidOperationException => 409,
            OperationCanceledException => 504,
            _ => 502
        };
        var detail = ex is OperationCanceledException ? "Ollama не успела ответить. Увеличьте Rag:TimeoutSeconds."
            : ex is HttpRequestException ? $"Не удалось получить ответ Ollama. Проверьте адрес и установленные модели. {ex.Message}"
            : ex.Message;
        await Results.Problem(statusCode: status, detail: detail).ExecuteAsync(context);
    }
});
app.MapGet("/", () => Results.Ok(new
{
    service = "Простой RAG + Ollama",
    endpoints = new[] { "GET /api/status", "POST /api/reindex", "POST /api/documents", "POST /api/ask" }
}));
app.MapGet("/api/status", (RagService rag) => Results.Ok(rag.Status()));
app.MapPost("/api/reindex", async (RagService rag, CancellationToken ct) =>
    Results.Ok(await rag.ReindexAsync(ct)));
app.MapPost("/api/documents", async (DocumentRequest request, RagService rag, CancellationToken ct) =>
    Results.Ok(await rag.AddDocumentAsync(request.Name, request.Text, ct)));
app.MapPost("/api/ask", async (AskRequest request, RagService rag, CancellationToken ct) =>
    Results.Ok(await rag.AskAsync(request.Question, request.TopK, ct)));
app.Run();

public record DocumentRequest(string Name, string Text);
public record AskRequest(string Question, int? TopK = null);
