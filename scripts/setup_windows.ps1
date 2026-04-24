param(
    [string]$PythonCommand = "",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Resolve-PythonCommand {
    param([string]$RequestedCommand)

    if ($RequestedCommand) {
        return $RequestedCommand
    }

    $candidates = @(
        'py -3.12',
        'py -3.11',
        'py -3.10',
        'py -3',
        'python'
    )

    foreach ($candidate in $candidates) {
        try {
            $parts = $candidate.Split(" ")
            $exe = $parts[0]
            $args = @()
            if ($parts.Length -gt 1) {
                $args = $parts[1..($parts.Length - 1)]
            }
            & $exe @args --version *> $null
            if ($LASTEXITCODE -eq 0 -and $exe) {
                return $candidate
            }
        } catch {
        }
    }

    throw "未找到可用的 Python。请先安装 Python 3.10+，并确保 py 或 python 命令可用。"
}

$pythonCmd = Resolve-PythonCommand -RequestedCommand $PythonCommand
$venvPath = Join-Path $repoRoot $VenvDir

Write-Host "==> 使用 Python 命令: $pythonCmd"

if (-not (Test-Path $venvPath)) {
    Write-Host "==> 创建虚拟环境: $venvPath"
    $pythonParts = $pythonCmd.Split(" ")
    $pythonExe = $pythonParts[0]
    $pythonArgs = @()
    if ($pythonParts.Length -gt 1) {
        $pythonArgs = $pythonParts[1..($pythonParts.Length - 1)]
    }
    & $pythonExe @pythonArgs -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "创建虚拟环境失败。"
    }
} else {
    Write-Host "==> 复用已有虚拟环境: $venvPath"
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "虚拟环境创建不完整，未找到 $venvPython"
}

Write-Host "==> 升级 pip/setuptools/wheel"
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "升级 pip 失败。"
}

Write-Host "==> 安装项目依赖（editable 模式）"
& $venvPython -m pip install -e .
if ($LASTEXITCODE -ne 0) {
    throw "安装项目依赖失败。"
}

Write-Host "==> 校验关键依赖"
& $venvPython -c "import matplotlib, numpy, scipy, yaml; print('Python environment is ready.')"
if ($LASTEXITCODE -ne 0) {
    throw "关键依赖校验失败。"
}

Write-Host ""
Write-Host "环境配置完成。"
Write-Host "激活命令: $VenvDir\Scripts\activate"
Write-Host "运行示例: python examples\run_from_config.py configs\pusch_awgn.yaml"
