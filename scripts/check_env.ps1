<#
.SYNOPSIS
  检查并可选安装本项目所需的 Python 依赖（针对 Windows PowerShell，项目结构假定 scripts/ 在仓库根下）。

.DESCRIPTION
  - 激活仓库内的 .venv（若存在），可选创建 venv 并安装 requirements.txt 中的依赖。
  - 在激活环境后，逐个测试关键包的导入（pandas, numpy, sklearn, chardet, torch），并输出结果。

.PARAMETER Install
  若指定，尝试使用 `.venv\Scripts\pip.exe` 安装 `requirements.txt`（若存在），或安装缺失的常用包。

.PARAMETER CreateVenv
  若指定且 `.venv` 不存在，尝试使用系统 python 创建虚拟环境（`python -m venv .venv`）。

EXAMPLE
  # 仅检查环境（不安装）
  .\check_env.ps1

  # 创建 venv（如果不存在）并安装 requirements.txt
  .\check_env.ps1 -CreateVenv -Install
#>

param(
    [switch]$Install,
    [switch]$CreateVenv,
    [string]$Requirements = "..\requirements.txt"
)

Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$venvPath = Join-Path $repoRoot ".venv"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

function Write-Ok($msg) { Write-Host "[OK]" -ForegroundColor Green; Write-Host " $msg" }
function Write-Warn($msg) { Write-Host "[WARN]" -ForegroundColor Yellow; Write-Host " $msg" }
function Write-Err($msg) { Write-Host "[ERR]" -ForegroundColor Red; Write-Host " $msg" }

Write-Host "Repository root: $repoRoot"

if (-not (Test-Path $venvPath)) {
    if ($CreateVenv) {
        Write-Host "虚拟环境 .venv 不存在，尝试使用系统 python 创建..."
        try {
            & python -m venv "$venvPath"
            Write-Ok "已创建 .venv"
        }
        catch {
            Write-Err "创建 .venv 失败：$($_.Exception.Message) 。请手动运行 'python -m venv .venv' 并重试。"
            exit 1
        }
    }
    else {
        Write-Warn ".venv 未找到。若希望自动创建请添加 -CreateVenv 开关。继续检查系统环境..."
    }
}

if (Test-Path $activateScript) {
    Write-Host "激活虚拟环境： $activateScript"
    try {
        # PowerShell 激活脚本需 dot-source
        . $activateScript
        Write-Ok "已激活 .venv"
    }
    catch {
        Write-Err "激活 .venv 失败：$($_.Exception.Message)"
    }
}
else {
    Write-Warn "激活脚本未找到，虚拟环境可能未安装或路径不正确： $activateScript"
}

if ($Install) {
    if (Test-Path (Resolve-Path (Join-Path $repoRoot $Requirements) -ErrorAction SilentlyContinue)) {
        $reqPath = Resolve-Path (Join-Path $repoRoot $Requirements)
        Write-Host "使用 $venvPip 安装依赖： $reqPath"
        try {
            & $venvPip install -r $reqPath
            Write-Ok "依赖安装完成"
        }
        catch {
            Write-Err "依赖安装失败：$($_.Exception.Message)"
        }
    }
    else {
        Write-Warn "未找到 requirements.txt，尝试安装主要包（torch, numpy, pandas, scikit-learn, matplotlib, chardet）"
        try {
            & $venvPip install torch numpy pandas scikit-learn matplotlib chardet
            Write-Ok "主要包安装完成"
        }
        catch {
            Write-Err "主要包安装失败：$($_.Exception.Message)"
        }
    }
}

# 导入测试
$modules = @('pandas','numpy','sklearn','chardet','torch')
foreach ($m in $modules) {
    Write-Host "测试导入： $m"
    try {
        $cmd = "import $m; print('__OK__', getattr($m, '__version__', 'unknown') )"
        # 使用 venv python 以保证环境一致
        & $venvPython -c $cmd 2>&1 | ForEach-Object { Write-Host $_ }
    }
    catch {
        Write-Err "模块 $m 导入测试发生错误：$($_.Exception.Message)"
    }
}

Write-Host "\n完成。若在 VS Code 中仍看到 Pylance 报错："
Write-Host " - 确认已在 VS Code 中选择解释器：Ctrl+Shift+P -> Python: Select Interpreter -> 选择 ${repoRoot}\.venv\Scripts\python.exe"
Write-Host " - 选择后执行：Ctrl+Shift+P -> Developer: Reload Window 或 Python: Restart Language Server"

Write-Host "如果你希望我把此脚本注册到 README 或添加到 git hooks，请回复我将继续。"
