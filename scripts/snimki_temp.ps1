# Headless Edge shots of the temperature runs: every t<T>.html in the folder without its t<T>.png gets one
# (1280x860, 4 s of animation). -Page also shoots the built index.html tall for review.
#   powershell -File snimki_temp.ps1 <folder> [-Page]
param([string]$Dir, [switch]$Page)
$E = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
$prof = Join-Path $env:TEMP "edge_snimki_temp_1510"
function Shot($html, $png, $size) {
    $url = "file:///" + ($html -replace '\\', '/')
    $p = Start-Process -FilePath $E -PassThru -Wait -WindowStyle Hidden -ArgumentList @(
        "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--user-data-dir=$prof",
        "--window-size=$size", "--virtual-time-budget=4000", "--screenshot=$png", $url)
    "{0} -> {1} ({2})" -f (Split-Path $html -Leaf), (Split-Path $png -Leaf), $p.ExitCode
}
Get-ChildItem $Dir -Filter "t*.html" | ForEach-Object {
    $png = [IO.Path]::ChangeExtension($_.FullName, ".png")
    if (-not (Test-Path $png)) { Shot $_.FullName $png "1280,860" }
}
if ($Page) { Shot (Join-Path $Dir "index.html") (Join-Path $Dir "_stranica.png") "1300,3200" }
