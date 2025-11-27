$content = Get-Content "templates\partials\current_indicators.html" -Raw
$lines = $content -split "`r?`n"

# Rimuovi le righe 78-88 (array 0-based, quindi 77-87)
$newLines = $lines[0..76] + $lines[88..($lines.Length-1)]

$newContent = $newLines -join "`r`n" 
Set-Content "templates\partials\current_indicators.html" -Value $newContent -NoNewline
