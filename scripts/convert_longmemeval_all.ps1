$python = "D:\workspace\memory\aml-text-memory\.venv\Scripts\python.exe"
$converter = "D:\workspace\memory\aml-text-memory\scripts\convert_longmemeval.py"
$inputDir = "D:\workspace\memory\datasets\hf\longmemeval"
$outputDir = "D:\workspace\memory\aml-text-memory\examples"

& $python $converter --input (Join-Path $inputDir "longmemeval_s_cleaned.json") --output (Join-Path $outputDir "longmemeval_s_eval.jsonl")
Write-Output "S_DONE"

& $python $converter --input (Join-Path $inputDir "longmemeval_m_cleaned.json") --output (Join-Path $outputDir "longmemeval_m_eval.jsonl")
Write-Output "M_DONE"
