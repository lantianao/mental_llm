param(
    [string]$InputPath = "data_raw/counselbench_eval.csv",
    [string]$OutputDir = "data/dpo",
    [double]$MinScoreDiff = 1.0,
    [double]$TrainRatio = 0.8,
    [double]$DevRatio = 0.1,
    [int]$Seed = 42
)

function Clean-Text {
    param([AllowNull()][string]$Value)
    if ($null -eq $Value) {
        return ""
    }
    $Text = [System.Net.WebUtility]::HtmlDecode($Value)
    $Text = [regex]::Replace($Text, "<\s*br\s*/?\s*>", "`n", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    $Text = [regex]::Replace($Text, "<[^>]+>", " ")
    $Text = $Text -replace "`r`n", "`n"
    $Text = $Text -replace "`r", "`n"
    $Text = [regex]::Replace($Text, "[ `t]+", " ")
    $Text = [regex]::Replace($Text, "`n{3,}", "`n`n")
    return $Text.Trim()
}

function Format-Prompt {
    param([object]$Row)
    $Parts = New-Object System.Collections.Generic.List[string]
    $Title = Clean-Text $Row.questionTitle
    $Body = Clean-Text $Row.questionText
    if ($Title.Length -gt 0) {
        $Parts.Add("Title: $Title")
    }
    if ($Body.Length -gt 0) {
        $Parts.Add("Question:`n$Body")
    }
    $Parts.Add("Please provide a supportive, safe, and clinically cautious response.")
    return ($Parts -join "`n`n")
}

function Write-Jsonl {
    param(
        [string]$Path,
        [object]$Rows
    )
    $Parent = Split-Path -Parent $Path
    if ($Parent -and -not (Test-Path $Parent)) {
        New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    }
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    $Writer = New-Object System.IO.StreamWriter($Path, $false, $Utf8NoBom)
    try {
        foreach ($Row in @($Rows)) {
            $Writer.WriteLine(($Row | ConvertTo-Json -Compress -Depth 10))
        }
    }
    finally {
        $Writer.Close()
    }
}

function Get-Mean {
    param([double[]]$Values)
    if ($Values.Count -eq 0) {
        return $null
    }
    return (($Values | Measure-Object -Average).Average)
}

function Get-PopStd {
    param([double[]]$Values)
    if ($Values.Count -le 1) {
        return 0.0
    }
    $Mean = Get-Mean $Values
    $Sum = 0.0
    foreach ($Value in $Values) {
        $Sum += [Math]::Pow($Value - $Mean, 2)
    }
    return [Math]::Sqrt($Sum / $Values.Count)
}

function Shuffle-List {
    param(
        [object[]]$Items,
        [int]$RandomSeed
    )
    $Random = New-Object System.Random($RandomSeed)
    $Array = @($Items)
    for ($Index = $Array.Count - 1; $Index -gt 0; $Index--) {
        $SwapIndex = $Random.Next(0, $Index + 1)
        $Temp = $Array[$Index]
        $Array[$Index] = $Array[$SwapIndex]
        $Array[$SwapIndex] = $Temp
    }
    return $Array
}

if (-not (Test-Path $InputPath)) {
    throw "Input CSV not found: $InputPath"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Rows = Import-Csv -Path $InputPath -Encoding UTF8

$GroupedResponses = New-Object System.Collections.Generic.List[object]
foreach ($CsvGroup in ($Rows | Group-Object -Property questionID, responder | Sort-Object Name)) {
    $GroupRows = @($CsvGroup.Group)
    $First = $GroupRows[0]
    $Scores = @()
    foreach ($Row in $GroupRows) {
        $Parsed = 0.0
        if ([double]::TryParse($Row.overall_score, [ref]$Parsed)) {
            $Scores += $Parsed
        }
    }
    if ($Scores.Count -eq 0) {
        continue
    }

    $Mean = Get-Mean $Scores
    $Std = Get-PopStd $Scores
    $GroupedResponses.Add([pscustomobject]@{
        question_id = $First.questionID
        topic = $First.topic
        responder = $First.responder
        prompt = Format-Prompt $First
        response = Clean-Text $First.response
        mean_overall_score = [Math]::Round($Mean, 6)
        std_overall_score = [Math]::Round($Std, 6)
        num_annotations = $Scores.Count
        annotation_scores = @($Scores)
    })
}

$QuestionIds = @($GroupedResponses | Select-Object -ExpandProperty question_id -Unique | Sort-Object)
$QuestionIds = @(Shuffle-List -Items $QuestionIds -RandomSeed $Seed)
$TotalQuestions = $QuestionIds.Count
$TrainCount = [Math]::Round($TotalQuestions * $TrainRatio)
$DevCount = [Math]::Round($TotalQuestions * $DevRatio)

$SplitMap = @{}
for ($Index = 0; $Index -lt $QuestionIds.Count; $Index++) {
    $Split = "test"
    if ($Index -lt $TrainCount) {
        $Split = "train"
    }
    elseif ($Index -lt ($TrainCount + $DevCount)) {
        $Split = "dev"
    }
    $SplitMap[$QuestionIds[$Index]] = $Split
}

$Pairs = New-Object System.Collections.Generic.List[object]
foreach ($QuestionGroup in ($GroupedResponses | Group-Object -Property question_id | Sort-Object Name)) {
    $QuestionId = $QuestionGroup.Name
    $Responses = @($QuestionGroup.Group)
    for ($I = 0; $I -lt $Responses.Count; $I++) {
        for ($J = $I + 1; $J -lt $Responses.Count; $J++) {
            $Left = $Responses[$I]
            $Right = $Responses[$J]
            $Diff = [double]$Left.mean_overall_score - [double]$Right.mean_overall_score
            if ([Math]::Abs($Diff) -lt $MinScoreDiff) {
                continue
            }
            if ($Diff -gt 0) {
                $Chosen = $Left
                $Rejected = $Right
            }
            else {
                $Chosen = $Right
                $Rejected = $Left
            }
            $Pairs.Add([pscustomobject]@{
                question_id = $QuestionId
                topic = $Chosen.topic
                split = $SplitMap[$QuestionId]
                prompt = $Chosen.prompt
                chosen = $Chosen.response
                rejected = $Rejected.response
                chosen_responder = $Chosen.responder
                rejected_responder = $Rejected.responder
                chosen_score = [Math]::Round([double]$Chosen.mean_overall_score, 4)
                rejected_score = [Math]::Round([double]$Rejected.mean_overall_score, 4)
                score_diff = [Math]::Round(([double]$Chosen.mean_overall_score - [double]$Rejected.mean_overall_score), 4)
                preference_source = "mean_overall_score"
            })
        }
    }
}

Write-Jsonl -Path (Join-Path $OutputDir "grouped_responses.jsonl") -Rows $GroupedResponses.ToArray()
foreach ($Split in @("train", "dev", "test")) {
    $SplitPairs = @($Pairs | Where-Object { $_.split -eq $Split })
    Write-Jsonl -Path (Join-Path $OutputDir "dpo_pairs_$Split.jsonl") -Rows $SplitPairs
}

$QuestionSplitRows = foreach ($QuestionId in ($SplitMap.Keys | Sort-Object)) {
    [pscustomobject]@{
        question_id = $QuestionId
        split = $SplitMap[$QuestionId]
    }
}
($QuestionSplitRows | ConvertTo-Json -Depth 5) | Set-Content -Path (Join-Path $OutputDir "question_split.json") -Encoding UTF8

$Diffs = @($Pairs | Select-Object -ExpandProperty score_diff)
$PairQuestionIds = @($Pairs | Select-Object -ExpandProperty question_id -Unique)
$Report = [pscustomobject]@{
    min_score_diff = $MinScoreDiff
    num_grouped_question_responses = $GroupedResponses.Count
    num_questions = $TotalQuestions
    num_questions_with_pairs = $PairQuestionIds.Count
    question_split_counts = [pscustomobject]@{
        train = @($SplitMap.Values | Where-Object { $_ -eq "train" }).Count
        dev = @($SplitMap.Values | Where-Object { $_ -eq "dev" }).Count
        test = @($SplitMap.Values | Where-Object { $_ -eq "test" }).Count
    }
    num_pairs = $Pairs.Count
    pair_split_counts = [pscustomobject]@{
        train = @($Pairs | Where-Object { $_.split -eq "train" }).Count
        dev = @($Pairs | Where-Object { $_.split -eq "dev" }).Count
        test = @($Pairs | Where-Object { $_.split -eq "test" }).Count
    }
    pair_question_split_counts = [pscustomobject]@{
        train = @($Pairs | Where-Object { $_.split -eq "train" } | Select-Object -ExpandProperty question_id -Unique).Count
        dev = @($Pairs | Where-Object { $_.split -eq "dev" } | Select-Object -ExpandProperty question_id -Unique).Count
        test = @($Pairs | Where-Object { $_.split -eq "test" } | Select-Object -ExpandProperty question_id -Unique).Count
    }
    chosen_responder_counts = [pscustomobject]@{
        gemini = @($Pairs | Where-Object { $_.chosen_responder -eq "gemini" }).Count
        gpt4 = @($Pairs | Where-Object { $_.chosen_responder -eq "gpt4" }).Count
        human = @($Pairs | Where-Object { $_.chosen_responder -eq "human" }).Count
        llama3 = @($Pairs | Where-Object { $_.chosen_responder -eq "llama3" }).Count
    }
    rejected_responder_counts = [pscustomobject]@{
        gemini = @($Pairs | Where-Object { $_.rejected_responder -eq "gemini" }).Count
        gpt4 = @($Pairs | Where-Object { $_.rejected_responder -eq "gpt4" }).Count
        human = @($Pairs | Where-Object { $_.rejected_responder -eq "human" }).Count
        llama3 = @($Pairs | Where-Object { $_.rejected_responder -eq "llama3" }).Count
    }
    score_diff = [pscustomobject]@{
        min = if ($Diffs.Count -gt 0) { ($Diffs | Measure-Object -Minimum).Minimum } else { $null }
        max = if ($Diffs.Count -gt 0) { ($Diffs | Measure-Object -Maximum).Maximum } else { $null }
        mean = if ($Diffs.Count -gt 0) { ($Diffs | Measure-Object -Average).Average } else { $null }
    }
}

$ReportPath = Join-Path $OutputDir "dpo_pair_report.json"
$Report | ConvertTo-Json -Depth 10 | Set-Content -Path $ReportPath -Encoding UTF8
$Report | ConvertTo-Json -Depth 10
