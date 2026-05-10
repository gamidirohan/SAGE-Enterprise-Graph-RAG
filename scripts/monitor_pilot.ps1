# Monitor pilot progress in real time

$resultsFile = 'D:\College\Sem_8\SAGE-Enterprise-Graph-RAG\results\pilot_results.json'
$logFile = 'D:\College\Sem_8\SAGE-Enterprise-Graph-RAG\pilot_run.log'

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PILOT MONITORING DASHBOARD" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if results file exists
if (Test-Path $resultsFile) {
    Write-Host "RESULTS FILE CREATED" -ForegroundColor Green
    $item = Get-Item $resultsFile
    Write-Host "  Last modified: $($item.LastWriteTime)" -ForegroundColor Green
    
    try {
        $json = Get-Content $resultsFile -Raw | ConvertFrom-Json
        $count = @($json).Count
        if ($null -eq $count) { $count = 1 }
        Write-Host "  Queries processed: $count" -ForegroundColor Yellow
    } catch {
        Write-Host "  (JSON still being written...)" -ForegroundColor Gray
    }
} else {
    Write-Host "WAITING..." -ForegroundColor Yellow
    Write-Host "  Results file not yet created" -ForegroundColor Gray
    Write-Host "  (Pilot is initializing models)" -ForegroundColor Gray
}

Write-Host ""

# Check log file for last status
if (Test-Path $logFile) {
    Write-Host "LATEST LOG ENTRIES:" -ForegroundColor Cyan
    $lines = @(Get-Content $logFile -Tail 5)
    foreach ($line in $lines) {
        Write-Host "  $line" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "PILOT COMPLETION INDICATORS:" -ForegroundColor Cyan
Write-Host "  LOOK FOR: Saved pilot results to ..." -ForegroundColor Yellow
Write-Host "  OR: Pilot run complete" -ForegroundColor Yellow
Write-Host ""
Write-Host "TO CHECK PROGRESS:" -ForegroundColor Cyan
Write-Host "  Get-Content pilot_run.log -Tail 5" -ForegroundColor Gray
Write-Host ""
