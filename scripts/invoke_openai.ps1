$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = 'Stop'

try {
    $apiKey = if ($env:openai_api_key) { $env:openai_api_key } else { $env:OPENAI_API_KEY }
    $baseUrl = if ($env:base_url) { $env:base_url } else { $env:BASE_URL }
    if (-not $apiKey) { throw 'Missing OpenAI API key environment variable' }
    if (-not $baseUrl) { throw 'Missing base_url environment variable' }

    $uri = $baseUrl.TrimEnd('/')
    if ([Uri]$uri -and ([Uri]$uri).AbsolutePath -eq '/') { $uri += '/v1' }
    $uri += '/chat/completions'
    $encodedBody = [Console]::In.ReadToEnd().Trim()
    $body = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($encodedBody))
    $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($body)
    $headers = @{ Authorization = "Bearer $apiKey" }
    $webResponse = Invoke-WebRequest -UseBasicParsing -Uri $uri -Headers $headers -Method Post -ContentType 'application/json; charset=utf-8' -Body $bodyBytes -TimeoutSec 300
    $stream = $webResponse.RawContentStream
    $stream.Position = 0
    $buffer = New-Object System.IO.MemoryStream
    $stream.CopyTo($buffer)
    $jsonText = [System.Text.Encoding]::UTF8.GetString($buffer.ToArray())
    $response = $jsonText | ConvertFrom-Json
    $content = $response.choices[0].message.content
    if (-not $content) { throw 'Model returned empty content' }
    [Console]::Out.Write($content)
}
catch {
    [Console]::Error.Write(($_ | Out-String))
    exit 1
}
