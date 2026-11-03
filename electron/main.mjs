import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron'
import { spawn, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { StringDecoder } from 'node:string_decoder'
import { fileURLToPath } from 'node:url'
import { activate as activateLicense, requestCode as licenseRequestCode, status as licenseStatus } from './license.mjs'

const currentDir = path.dirname(fileURLToPath(import.meta.url))
const projectDir = path.resolve(currentDir, '..')
const isDev = process.argv.includes('--dev') || Boolean(process.env.VITE_DEV_SERVER_URL)

let mainWindow = null
let trainingProcess = null
let monitorProcess = null
let monitorTimer = null
let monitorStats = null
let lastCpuSnapshot = null
let diskCache = []
let monitorTicks = 0
let trainingLogs = []
let activeTrainingConfig = null
let trainingTelemetry = null
let taggerProcess = null
let taggerLogs = []

const settingsPath = () => path.join(app.getPath('userData'), 'preferences.json')

function readSettings() {
  try {
    return JSON.parse(fs.readFileSync(settingsPath(), 'utf8'))
  } catch {
    return {
      language: 'zh',
      theme: 'light',
      glass: true,
      trainerRoot: '',
    }
  }
}

function writeSettings(next) {
  fs.mkdirSync(path.dirname(settingsPath()), { recursive: true })
  fs.writeFileSync(settingsPath(), JSON.stringify(next, null, 2), 'utf8')
  return next
}

function candidateRoots() {
  const preferences = readSettings()
  const portableExecutable = process.env.PORTABLE_EXECUTABLE_FILE || ''
  const roots = [
    process.env.LORA_TRAINER_ROOT,
    preferences.trainerRoot,
    process.env.PORTABLE_EXECUTABLE_DIR,
    portableExecutable ? path.dirname(portableExecutable) : '',
    process.cwd(),
    path.dirname(process.execPath),
    path.resolve(projectDir, '..'),
    projectDir,
    app.isPackaged ? path.dirname(process.resourcesPath) : '',
  ].filter(Boolean)
  return [...new Set(roots.map((root) => path.resolve(root)))]
}

function firstExisting(candidates) {
  return candidates.find((entry) => entry && fs.existsSync(entry)) || ''
}

function findPython() {
  if (process.env.LORA_TRAINER_PYTHON && fs.existsSync(process.env.LORA_TRAINER_PYTHON)) {
    return process.env.LORA_TRAINER_PYTHON
  }

  const candidates = []
  for (const root of candidateRoots()) {
    if (process.platform === 'win32') {
      candidates.push(
        path.join(root, 'python_embeded_DP', 'python.exe'),
        path.join(root, 'python', 'python.exe'),
        path.join(root, '.venv', 'Scripts', 'python.exe'),
      )
    } else {
      candidates.push(
        path.join(root, 'python_embeded_DP', 'bin', 'python'),
        path.join(root, 'python', 'bin', 'python'),
        path.join(root, '.venv', 'bin', 'python'),
      )
    }
  }

  return firstExisting(candidates)
}

function findBackendFile(filename) {
  const candidates = []
  for (const root of candidateRoots()) {
    candidates.push(
      path.join(root, 'app', 'backend', filename),
      path.join(root, 'backend', filename),
      path.join(root, filename),
    )
  }
  // Python and other external programs cannot execute a path located inside
  // app.asar. Packaged builds therefore prefer real files beside the trainer
  // core (app/backend) or in app.asar.unpacked.
  if (app.isPackaged) {
    candidates.unshift(
      path.join(process.resourcesPath, 'app.asar.unpacked', 'backend', filename),
      path.join(process.resourcesPath, 'backend', filename),
    )
  } else {
    candidates.unshift(path.join(projectDir, 'backend', filename))
  }
  return firstExisting(candidates)
}

function findTrainerRoot() {
  const platformCore = process.platform === 'win32' ? 'core' : 'core_linux'
  return candidateRoots().find((root) => [
    path.join(root, 'app', 'backend', platformCore, 'train.py'),
    path.join(root, 'backend', platformCore, 'train.py'),
    path.join(root, 'app', 'backend', 'core', 'train.py'),
    path.join(root, 'backend', 'core', 'train.py'),
  ].some((entry) => fs.existsSync(entry))) || ''
}

function taggedPayload(output) {
  const matches = [...String(output || '').matchAll(/__JSON_START__(.*?)__JSON_END__/g)]
  if (!matches.length) return null
  try {
    return JSON.parse(matches.at(-1)[1])
  } catch {
    return null
  }
}

function send(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send(channel, payload)
}

function streamTaggedJson(processHandle, channel, onPlainLine, onTagged) {
  const streams = {
    stdout: { pending: '', decoder: new StringDecoder('utf8') },
    stderr: { pending: '', decoder: new StringDecoder('utf8') },
  }
  const processLine = (rawLine, stream) => {
    const line = rawLine.replace(/\x1B\][^\x07]*(?:\x07|\x1B\\)/g, '').replace(/[\u0000\u0008]/g, '').trimEnd()
    if (!line) return
    const match = line.match(/__JSON_START__(.*?)__JSON_END__/)
    if (match) {
      try {
        const payload = JSON.parse(match[1])
        onTagged?.(payload)
        send(channel, payload)
        return
      } catch {
        // Forward malformed messages to the log instead of hiding them.
      }
    }
    onPlainLine?.({ line, stream })
  }
  const consume = (chunk, stream = 'stdout') => {
    const state = streams[stream]
    state.pending += state.decoder.write(chunk)
    const lines = state.pending.split(/\r\n|[\n\r]/)
    state.pending = lines.pop() || ''
    for (const rawLine of lines) processLine(rawLine, stream)
  }
  const flush = (stream) => {
    const state = streams[stream]
    state.pending += state.decoder.end()
    if (state.pending) processLine(state.pending, stream)
    state.pending = ''
  }
  processHandle.stdout?.on('data', (chunk) => consume(chunk, 'stdout'))
  processHandle.stderr?.on('data', (chunk) => consume(chunk, 'stderr'))
  processHandle.stdout?.on('end', () => flush('stdout'))
  processHandle.stderr?.on('end', () => flush('stderr'))
}

function terminateProcess(processHandle) {
  if (!processHandle || processHandle.killed) return
  if (process.platform === 'win32' && processHandle.pid) {
    spawn('taskkill', ['/pid', String(processHandle.pid), '/T', '/F'], { windowsHide: true })
  } else if (processHandle.pid) {
    try {
      process.kill(-processHandle.pid, 'SIGTERM')
    } catch {
      processHandle.kill('SIGTERM')
    }
  }
}

function readCpuPercent() {
  const snapshot = os.cpus().reduce((sum, cpu) => {
    const total = Object.values(cpu.times).reduce((value, part) => value + part, 0)
    return { idle: sum.idle + cpu.times.idle, total: sum.total + total }
  }, { idle: 0, total: 0 })
  if (!lastCpuSnapshot) {
    lastCpuSnapshot = snapshot
    return 0
  }
  const idle = snapshot.idle - lastCpuSnapshot.idle
  const total = snapshot.total - lastCpuSnapshot.total
  lastCpuSnapshot = snapshot
  return total > 0 ? Math.max(0, Math.min(100, Number(((1 - idle / total) * 100).toFixed(1)))) : 0
}

function readNvidiaGpus() {
  const query = '--query-gpu=index,name,utilization.gpu,memory.total,memory.used,temperature.gpu'
  const executables = process.platform === 'win32'
    ? [
        'nvidia-smi.exe',
        path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'nvidia-smi.exe'),
        path.join(process.env.ProgramW6432 || process.env.ProgramFiles || 'C:\\Program Files', 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe'),
      ]
    : ['nvidia-smi']
  let result = null
  for (const executable of [...new Set(executables)]) {
    result = spawnSync(executable, [query, '--format=csv,noheader,nounits'], { encoding: 'utf8', windowsHide: true, timeout: 4000 })
    if (result.status === 0 && result.stdout) break
  }
  if (result.status !== 0 || !result.stdout) return []
  return result.stdout.trim().split(/\r?\n/).map((line) => {
    const values = line.split(',').map((value) => value.trim())
    if (values.length < 6) return null
    const total = Number(values[3]) * 1024 * 1024
    const used = Number(values[4]) * 1024 * 1024
    return { id: Number(values[0]), name: values[1], gpu_util: Number(values[2]), mem_total: total,
      mem_used: used, mem_free: total - used, temperature: Number(values[5]), source: 'nvidia-smi' }
  }).filter(Boolean)
}

function readWindowsDisks() {
  if (process.platform !== 'win32') return []
  const command = 'Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | Select-Object DeviceID,Size,FreeSpace | ConvertTo-Json -Compress'
  const result = spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', command], { encoding: 'utf8', windowsHide: true, timeout: 5000 })
  if (result.status !== 0 || !result.stdout.trim()) return diskCache
  try {
    const parsed = JSON.parse(result.stdout.trim())
    return (Array.isArray(parsed) ? parsed : [parsed]).map((disk) => {
      const total = Number(disk.Size) || 0
      const free = Number(disk.FreeSpace) || 0
      return { device: disk.DeviceID, mountpoint: `${disk.DeviceID}\\`, total, free, used: total - free,
        percent: total ? Number((((total - free) / total) * 100).toFixed(1)) : 0 }
    })
  } catch {
    return diskCache
  }
}

function collectNativeStats() {
  if (monitorTicks++ % 5 === 0) diskCache = readWindowsDisks()
  const total = os.totalmem()
  const available = os.freemem()
  const stats = {
    cpu_model: os.cpus()[0]?.model?.trim() || 'Unknown CPU', cpu_percent: readCpuPercent(),
    memory: { total, available, used: total - available, percent: total ? Number((((total - available) / total) * 100).toFixed(1)) : 0 },
    disks: diskCache, gpus: readNvidiaGpus(), timestamp: Date.now() / 1000,
  }
  monitorStats = stats
  send('monitor:stats', stats)
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1580,
    icon: path.join(projectDir, 'dist', 'profile.jpg'),
    height: 1000,
    minWidth: 1120,
    minHeight: 720,
    frame: false,
    titleBarStyle: 'hidden',
    backgroundColor: '#dff5ff',
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      preload: path.join(currentDir, 'preload.mjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  })

  mainWindow.once('ready-to-show', () => mainWindow?.show())
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) shell.openExternal(url)
    return { action: 'deny' }
  })

  const devUrl = process.env.VITE_DEV_SERVER_URL || 'http://127.0.0.1:5173'
  if (isDev) mainWindow.loadURL(devUrl)
  else mainWindow.loadFile(path.join(projectDir, 'dist', 'index.html'))
}

app.whenReady().then(() => {
  if (process.platform === 'win32') app.setAppUserModelId('com.exceptional.loratrainer')
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  terminateProcess(trainingProcess)
  terminateProcess(taggerProcess)
  terminateProcess(monitorProcess)
  if (monitorTimer) clearInterval(monitorTimer)
  if (process.platform !== 'darwin') app.quit()
})

ipcMain.on('window:minimize', () => mainWindow?.minimize())
ipcMain.on('window:toggle-maximize', () => {
  if (!mainWindow) return
  mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize()
})
ipcMain.on('window:close', () => mainWindow?.close())

ipcMain.handle('dialog:folder', async (_event, options = {}) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: options.title || '选择文件夹',
    defaultPath: options.defaultPath,
    properties: ['openDirectory', 'createDirectory'],
  })
  return result.canceled ? '' : result.filePaths[0]
})

ipcMain.handle('dialog:file', async (_event, options = {}) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: options.title || '选择文件',
    defaultPath: options.defaultPath,
    filters: options.filters,
    properties: ['openFile'],
  })
  return result.canceled ? '' : result.filePaths[0]
})

ipcMain.handle('dataset:scan', (_event, rootPath, captionExtension = '.txt') => {
  if (!rootPath || !fs.existsSync(rootPath)) return { success: false, pairs: 0, message: '目录不存在' }
  const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff'])
  const captionExt = String(captionExtension || '.txt').trim().toLowerCase()
  const normalizedCaptionExt = captionExt.startsWith('.') ? captionExt : `.${captionExt}`
  let pairs = 0
  try {
    const pending = [rootPath]
    while (pending.length) {
      const directory = pending.pop()
      const entries = fs.readdirSync(directory, { withFileTypes: true })
      const captions = new Set(entries.filter((entry) => entry.isFile() && path.extname(entry.name).toLowerCase() === normalizedCaptionExt).map((entry) => path.basename(entry.name, normalizedCaptionExt).toLowerCase()))
      for (const entry of entries) {
        if (entry.isDirectory() && entry.name.toLowerCase() !== 'cache') pending.push(path.join(directory, entry.name))
        else if (entry.isFile() && imageExtensions.has(path.extname(entry.name).toLowerCase()) && captions.has(path.basename(entry.name, path.extname(entry.name)).toLowerCase())) pairs += 1
      }
    }
    return { success: true, pairs }
  } catch (error) {
    return { success: false, pairs: 0, message: error.message }
  }
})

ipcMain.handle('preferences:get', () => readSettings())
ipcMain.handle('preferences:set', (_event, patch) => writeSettings({ ...readSettings(), ...patch }))

ipcMain.handle('system:open-external', async (_event, url) => {
  if (!/^https?:\/\//i.test(url)) return { success: false, error: 'Unsupported URL' }
  await shell.openExternal(url)
  return { success: true }
})

ipcMain.handle('system:reveal', (_event, targetPath) => {
  if (!targetPath) return { success: false }
  shell.showItemInFolder(targetPath)
  return { success: true }
})

ipcMain.handle('system:open-path', async (_event, targetPath) => {
  if (!targetPath || !fs.existsSync(targetPath)) return { success: false, message: '目录尚未生成' }
  const message = await shell.openPath(targetPath)
  return message ? { success: false, message } : { success: true }
})

ipcMain.handle('system:diagnostics', () => {
  const python = findPython()
  const bridge = findBackendFile('backend_bridge.py')
  const monitor = findBackendFile('monitor.py')
  const trainerRoot = findTrainerRoot()
  const pythonProbe = python ? spawnSync(python, ['--version'], { encoding: 'utf8', windowsHide: true }) : null
  const pythonVersion = pythonProbe ? `${pythonProbe.stdout || ''}${pythonProbe.stderr || ''}`.trim() : ''
  return {
    appVersion: app.getVersion(),
    platform: `${process.platform} ${process.arch}`,
    python,
    pythonVersion,
    pythonReady: pythonProbe?.status === 0,
    pythonDiagnostic: python ? (pythonProbe?.status === 0 ? '' : '整合包内 Python 无法启动。') : '未找到 python_embeded_DP\\python.exe。请把 EXE 放在原训练整合包根目录，或在环境诊断中选择该根目录。',
    bridge,
    bridgeReady: Boolean(bridge),
    monitor,
    monitorReady: Boolean(monitor),
    trainerRoot,
    trainerReady: Boolean(trainerRoot),
    trainerRoots: candidateRoots(),
  }
})

ipcMain.handle('training:start', async (_event, config) => {
  const license = licenseStatus(app.getPath('userData'))
  if (!license.active) return { success: false, message: `需要有效离线许可证：${license.reason || '未激活'}` }
  if (trainingProcess) return { success: false, message: '训练已经在运行' }
  const python = findPython()
  if (!python) return { success: false, message: '未找到原整合包的 python_embeded_DP\\python.exe。请在“环境诊断”中选择整合包根目录后重试。' }
  const bridge = findBackendFile('backend_bridge.py')
  if (!bridge) return { success: false, message: '未找到 backend_bridge.py，请在环境诊断中设置整合包目录。' }
  const trainerRoot = findTrainerRoot()
  if (!trainerRoot) return { success: false, message: '未找到训练后端。请在“环境诊断”中选择包含 app/backend/core/train.py 的原整合包根目录。' }

  activeTrainingConfig = config
  trainingLogs = []
  trainingTelemetry = { startedAt: Date.now(), currentStep: 0, totalSteps: 0, samples: [], outputDir: '' }
  const serializedConfig = JSON.stringify(config)
  const validation = spawnSync(python, [bridge, '--action', 'validate_config', '--full_config', serializedConfig, '--trainer_root', trainerRoot], {
    cwd: path.dirname(bridge), encoding: 'utf8', windowsHide: true, timeout: 120000,
    env: { ...process.env, LORA_TRAINER_ROOT: trainerRoot, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
  })
  const validationPayload = taggedPayload(`${validation.stdout || ''}\n${validation.stderr || ''}`)
  if (validation.status !== 0 || !validationPayload?.success) {
    return { success: false, message: validationPayload?.message || validation.error?.message || '训练配置校验失败。' }
  }
  const captionScan = validationPayload.captionScan
  if (captionScan?.applicable) {
    send('training:status-event', { status: 'CAPTION_SCAN', ...captionScan })
  }
  const args = [bridge, '--action', 'start_training', '--full_config', serializedConfig, '--trainer_root', trainerRoot]
  trainingProcess = spawn(python, args, {
    cwd: path.dirname(bridge),
    detached: process.platform !== 'win32',
    windowsHide: true,
    env: {
      ...process.env,
      LORA_TRAINER_ROOT: trainerRoot,
      PYTHONUTF8: '1',
      PYTHONIOENCODING: 'utf-8',
      PYTHONUNBUFFERED: '1',
    },
  })

  streamTaggedJson(trainingProcess, 'training:status-event', ({ line, stream }) => {
    const payload = { line, stream, timestamp: Date.now() }
    trainingLogs.push(payload)
    send('training:output', payload)
  }, (payload) => {
    if (payload.status === 'LOG') {
      const logPayload = { line: payload.line, stream: payload.kind === 'error' ? 'stderr' : 'stdout', kind: payload.kind, encoding: payload.encoding, timestamp: Date.now() }
      trainingLogs.push(logPayload)
      send('training:output', logPayload)
    }
    if (payload.status === 'TRAINING_STARTED' && trainingTelemetry) {
      trainingTelemetry.totalSteps = Number(payload.total_steps) || 0
      trainingTelemetry.outputDir = payload.output_dir || ''
    }
    if (payload.status === 'TRAINING_PLAN' && trainingTelemetry) {
      trainingTelemetry.totalSteps = Number(payload.totalSteps) || 0
      trainingTelemetry.estimated = false
    }
    if (payload.status === 'RAW_LOG_READY' && trainingTelemetry) {
      trainingTelemetry.rawLogPath = payload.rawLogPath || ''
      trainingTelemetry.outputDir = payload.outputDir || trainingTelemetry.outputDir
    }
    if (payload.status === 'PROGRESS' && trainingTelemetry) {
      Object.assign(trainingTelemetry, {
        currentStep: Number(payload.currentStep) || trainingTelemetry.currentStep,
        totalSteps: Number(payload.totalSteps) || trainingTelemetry.totalSteps,
        stepSeconds: Number(payload.stepSeconds), elapsedSeconds: Number(payload.elapsedSeconds),
        remainingSeconds: Number(payload.remainingSeconds), loss: Number(payload.loss),
        learningRate: payload.learningRate, epoch: payload.epoch, estimated: false,
        outputDir: payload.outputDir || trainingTelemetry.outputDir,
        rawLogPath: payload.rawLogPath || trainingTelemetry.rawLogPath,
      })
      payload.checkpointInterval = Number(activeTrainingConfig?.training?.save_every_n_steps) || 0
    }
  })

  trainingProcess.once('spawn', () => {
    send('training:status-event', { status: 'RUNNING', pid: trainingProcess?.pid })
  })
  trainingProcess.once('error', (error) => {
    send('training:status-event', { status: 'ERROR', message: error.message })
    trainingProcess = null
  })
  trainingProcess.once('close', (code) => {
    send('training:status-event', { status: code === 0 ? 'COMPLETED' : 'FAILED', code })
    trainingProcess = null
  })

  return { success: true, pid: trainingProcess.pid }
})

ipcMain.handle('training:preview', async (_event, config) => {
  const python = findPython()
  const bridge = findBackendFile('backend_bridge.py')
  const trainerRoot = findTrainerRoot()
  if (!python || !bridge || !trainerRoot) return { success: false, message: '未找到原整合包 Python 或训练核心，请先完成环境诊断。' }
  const result = spawnSync(python, [bridge, '--action', 'preview_plan', '--full_config', JSON.stringify(config), '--trainer_root', trainerRoot], {
    cwd: path.dirname(bridge), encoding: 'utf8', windowsHide: true, timeout: 120000,
    env: { ...process.env, LORA_TRAINER_ROOT: trainerRoot, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
  })
  const payload = taggedPayload(`${result.stdout || ''}\n${result.stderr || ''}`)
  return payload?.success ? payload : { success: false, message: payload?.message || result.error?.message || '无法生成训练计划。' }
})

ipcMain.handle('training:stop', () => {
  if (!trainingProcess) return { success: false, message: '当前没有训练任务' }
  terminateProcess(trainingProcess)
  trainingProcess = null
  send('training:status-event', { status: 'STOPPED' })
  return { success: true }
})

ipcMain.handle('training:status', () => ({
  running: Boolean(trainingProcess),
  pid: trainingProcess?.pid,
  logs: trainingLogs,
  config: activeTrainingConfig,
  telemetry: trainingTelemetry,
}))

ipcMain.handle('tagger:scan', (_event, rootPath) => {
  if (!rootPath || !fs.existsSync(rootPath)) return { success: false, total: 0, message: '目录不存在' }
  const extensions = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'])
  let total = 0
  const pending = [rootPath]
  try {
    while (pending.length) {
      const directory = pending.pop()
      for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
        const target = path.join(directory, entry.name)
        if (entry.isDirectory()) pending.push(target)
        else if (entry.isFile() && extensions.has(path.extname(entry.name).toLowerCase())) total += 1
      }
    }
    return { success: true, total }
  } catch (error) {
    return { success: false, total: 0, message: error.message }
  }
})

ipcMain.handle('tagger:diagnostics', () => {
  const python = findPython()
  const script = findBackendFile('wd14_tagger.py')
  if (!python || !script) return { success: false, message: '未找到 WD14 所需的整合包 Python 或后端脚本。' }
  const result = spawnSync(python, [script, '--diagnose'], { cwd: path.dirname(script), encoding: 'utf8', windowsHide: true, timeout: 30000, env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' } })
  const payload = taggedPayload(`${result.stdout || ''}\n${result.stderr || ''}`)
  return payload ? { success: true, ...payload } : { success: false, message: result.error?.message || '无法读取 WD14 运行环境。' }
})

ipcMain.handle('tagger:start', async (_event, config) => {
  const license = licenseStatus(app.getPath('userData'))
  if (!license.active) return { success: false, message: `需要有效离线许可证：${license.reason || '未激活'}` }
  if (taggerProcess) return { success: false, message: 'WD14 正在运行' }
  if (trainingProcess) return { success: false, message: '请先停止训练任务，再运行 WD14 打标。' }
  const python = findPython()
  if (!python) return { success: false, message: '未找到原整合包的 python_embeded_DP\\python.exe，无法运行 WD14。' }
  const script = findBackendFile('wd14_tagger.py')
  if (!script) return { success: false, message: '未找到 wd14_tagger.py' }
  const args = [script, '--root', config.rootPath, '--model-dir', path.join(app.getPath('userData'), 'wd14-models'), '--threshold', String(config.threshold), '--character-threshold', String(config.characterThreshold), '--trigger-words', config.triggerWords || '', '--blocked-words', config.blockedWords || '']
  if (config.overwrite) args.push('--overwrite')
  if (config.replaceUnderscores) args.push('--replace-underscores')
  taggerLogs = []
  taggerProcess = spawn(python, args, { cwd: path.dirname(script), windowsHide: true, env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' } })
  streamTaggedJson(taggerProcess, 'tagger:status-event', ({ line, stream }) => {
    const payload = { line, stream, timestamp: Date.now() }
    taggerLogs.push(payload)
    if (taggerLogs.length > 1000) taggerLogs.shift()
    send('tagger:output', payload)
  })
  taggerProcess.once('error', (error) => { send('tagger:status-event', { status: 'ERROR', message: error.message }); taggerProcess = null })
  taggerProcess.once('close', (code) => { send('tagger:status-event', { status: code === 0 ? 'COMPLETED' : 'FAILED', code }); taggerProcess = null })
  return { success: true, pid: taggerProcess.pid }
})

ipcMain.handle('license:status', () => licenseStatus(app.getPath('userData')))
ipcMain.handle('license:request-code', () => ({ requestCode: licenseRequestCode(app.getPath('userData')) }))
ipcMain.handle('license:activate', (_event, text) => activateLicense(String(text || ''), app.getPath('userData')))
ipcMain.handle('license:activate-file', (_event, filename) => {
  try { return activateLicense(fs.readFileSync(String(filename), 'utf8'), app.getPath('userData')) } catch (error) { return { active: false, reason: `无法读取许可证文件：${error.message}` } }
})

ipcMain.handle('tagger:stop', () => {
  if (!taggerProcess) return { success: false, message: 'WD14 当前未运行' }
  terminateProcess(taggerProcess)
  taggerProcess = null
  send('tagger:status-event', { status: 'STOPPED' })
  return { success: true }
})

ipcMain.handle('monitor:start', () => {
  if (monitorProcess || monitorTimer) return { success: true, message: 'already-running' }
  lastCpuSnapshot = null
  monitorTicks = 0
  const python = findPython()
  const script = findBackendFile('monitor.py')
  if (python && script) {
    try {
      monitorProcess = spawn(python, [script], {
        cwd: path.dirname(script), windowsHide: true,
        env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
      })
      streamTaggedJson(monitorProcess, 'monitor:stats', null, (payload) => {
        if (!payload?.error) monitorStats = payload
      })
      monitorProcess.once('error', () => {
        monitorProcess = null
        if (!monitorTimer) {
          collectNativeStats()
          monitorTimer = setInterval(collectNativeStats, 1000)
        }
      })
      monitorProcess.once('close', () => { monitorProcess = null })
      return { success: true, source: 'python-nvml' }
    } catch {
      monitorProcess = null
    }
  }
  collectNativeStats()
  monitorTimer = setInterval(collectNativeStats, 1000)
  return { success: true, source: 'native-fallback' }
})

ipcMain.handle('monitor:stop', () => {
  terminateProcess(monitorProcess)
  monitorProcess = null
  if (monitorTimer) clearInterval(monitorTimer)
  monitorTimer = null
  return { success: true }
})

ipcMain.handle('monitor:status', () => ({ running: Boolean(monitorProcess || monitorTimer), stats: monitorStats }))

// Compatibility handlers used by the original renderer.
ipcMain.handle('dialog:openFile', async (_event, options) => dialog.showOpenDialog(mainWindow, options))
ipcMain.handle('get-language', () => readSettings().language || 'zh')
ipcMain.handle('set-language', (_event, language) => writeSettings({ ...readSettings(), language }))
ipcMain.handle('get-theme', () => readSettings().theme || 'light')
ipcMain.handle('set-theme', (_event, theme) => writeSettings({ ...readSettings(), theme }))
ipcMain.handle('get-training-status', () => ({ running: Boolean(trainingProcess), logs: trainingLogs.map((item) => item.line) }))
ipcMain.handle('stop-training', () => {
  if (!trainingProcess) return { success: false, message: 'No training running' }
  terminateProcess(trainingProcess)
  trainingProcess = null
  return { success: true }
})
