import { contextBridge, ipcRenderer } from 'electron'

const listen = (channel, callback) => {
  const handler = (_event, payload) => callback(payload)
  ipcRenderer.on(channel, handler)
  return () => ipcRenderer.removeListener(channel, handler)
}

contextBridge.exposeInMainWorld('loraDesktop', {
  platform: process.platform,
  window: {
    minimize: () => ipcRenderer.send('window:minimize'),
    toggleMaximize: () => ipcRenderer.send('window:toggle-maximize'),
    close: () => ipcRenderer.send('window:close'),
  },
  dialog: {
    folder: (options = {}) => ipcRenderer.invoke('dialog:folder', options),
    file: (options = {}) => ipcRenderer.invoke('dialog:file', options),
  },
  dataset: {
    scan: (rootPath, captionExtension) => ipcRenderer.invoke('dataset:scan', rootPath, captionExtension),
  },
  preferences: {
    get: () => ipcRenderer.invoke('preferences:get'),
    set: (patch) => ipcRenderer.invoke('preferences:set', patch),
  },
  system: {
    openExternal: (url) => ipcRenderer.invoke('system:open-external', url),
    reveal: (targetPath) => ipcRenderer.invoke('system:reveal', targetPath),
    openPath: (targetPath) => ipcRenderer.invoke('system:open-path', targetPath),
    diagnostics: () => ipcRenderer.invoke('system:diagnostics'),
  },
  tagger: {
    diagnostics: () => ipcRenderer.invoke('tagger:diagnostics'),
    scan: (rootPath) => ipcRenderer.invoke('tagger:scan', rootPath),
    start: (config) => ipcRenderer.invoke('tagger:start', config),
    stop: () => ipcRenderer.invoke('tagger:stop'),
    onOutput: (callback) => listen('tagger:output', callback),
    onStatus: (callback) => listen('tagger:status-event', callback),
  },
  license: {
    status: () => ipcRenderer.invoke('license:status'),
    requestCode: () => ipcRenderer.invoke('license:request-code'),
    activate: (text) => ipcRenderer.invoke('license:activate', text),
    activateFile: (filename) => ipcRenderer.invoke('license:activate-file', filename),
  },
  training: {
    preview: (config) => ipcRenderer.invoke('training:preview', config),
    start: (config) => ipcRenderer.invoke('training:start', config),
    stop: () => ipcRenderer.invoke('training:stop'),
    status: () => ipcRenderer.invoke('training:status'),
    onOutput: (callback) => listen('training:output', callback),
    onStatus: (callback) => listen('training:status-event', callback),
  },
  monitor: {
    start: () => ipcRenderer.invoke('monitor:start'),
    stop: () => ipcRenderer.invoke('monitor:stop'),
    status: () => ipcRenderer.invoke('monitor:status'),
    onStats: (callback) => listen('monitor:stats', callback),
  },
})

// Keep the original application's IPC surface available for incremental migration.
contextBridge.exposeInMainWorld('ipcRenderer', {
  on(channel, callback) {
    const handler = (event, ...args) => callback(event, ...args)
    ipcRenderer.on(channel, handler)
    return () => ipcRenderer.removeListener(channel, handler)
  },
  off: (channel, callback) => ipcRenderer.removeListener(channel, callback),
  send: (channel, ...args) => ipcRenderer.send(channel, ...args),
  invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel),
})
