$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = 'Stop'
$client = $null
$content = $null

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

    # Windows PowerShell 5.1 may otherwise negotiate an obsolete TLS version or
    # reuse an unstable legacy HttpWebRequest connection. HttpClient plus TLS 1.2
    # is substantially more reliable with OpenAI-compatible HTTPS gateways.
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    [Net.ServicePointManager]::Expect100Continue = $false
    Add-Type -AssemblyName System.Net.Http
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.AutomaticDecompression = (
        [System.Net.DecompressionMethods]::GZip -bor
        [System.Net.DecompressionMethods]::Deflate
    )
    $client = New-Object System.Net.Http.HttpClient -ArgumentList $handler
    $client.Timeout = [TimeSpan]::FromSeconds(300)
    $client.DefaultRequestHeaders.Authorization = New-Object System.Net.Http.Headers.AuthenticationHeaderValue -ArgumentList 'Bearer', $apiKey
    $content = New-Object System.Net.Http.ByteArrayContent -ArgumentList (,$bodyBytes)
    $content.Headers.ContentType = New-Object System.Net.Http.Headers.MediaTypeHeaderValue -ArgumentList 'application/json'
    $content.Headers.ContentType.CharSet = 'utf-8'

    $httpResponse = $client.PostAsync($uri, $content).GetAwaiter().GetResult()
    $jsonText = $httpResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if (-not $httpResponse.IsSuccessStatusCode) {
        $detail = if ($jsonText.Length -gt 1200) { $jsonText.Substring(0, 1200) } else { $jsonText }
        throw "HTTP $([int]$httpResponse.StatusCode) $($httpResponse.ReasonPhrase): $detail"
    }
    $response = $jsonText | ConvertFrom-Json
    $content = $response.choices[0].message.content
    if (-not $content) { throw 'Model returned empty content' }
    [Console]::Out.Write($content)
}
catch {
    [Console]::Error.Write(($_ | Out-String))
    exit 1
}
finally {
    if ($content) { $content.Dispose() }
    if ($client) { $client.Dispose() }
}
