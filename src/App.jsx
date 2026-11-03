import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  ArrowRight,
  BadgeCheck,
  BookOpenText,
  Box,
  Check,
  ChevronDown,
  ChevronRight,
  CircleStop,
  Clock3,
  Cloud,
  Cpu,
  Database,
  ExternalLink,
  FileCode2,
  FileImage,
  Folder,
  FolderOpen,
  Gauge,
  Github,
  Globe2,
  HardDrive,
  Home,
  Info,
  Languages,
  Layers3,
  Maximize2,
  MemoryStick,
  Minimize2,
  Moon,
  MoreHorizontal,
  PanelLeftClose,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Square,
  Sun,
  Tags,
  TerminalSquare,
  Thermometer,
  Trash2,
  WandSparkles,
  X,
  Zap,
} from 'lucide-react'
import { MODEL_ARCHITECTURES, getModelProfile } from '../electron/model-profiles.mjs'

const desktop = window.loraDesktop
const GOOFISH_URL = 'https://www.goofish.com/personal?spm=a21ybx.search.nav.1.37db1d0atvj0bs'

const RATIOS = [
  { id: '1:1', w: 1, h: 1, zh: '方形', en: 'Square' },
  { id: '4:5', w: 4, h: 5, zh: '竖幅', en: 'Portrait' },
  { id: '3:4', w: 3, h: 4, zh: '经典竖幅', en: 'Classic portrait' },
  { id: '2:3', w: 2, h: 3, zh: '海报', en: 'Poster' },
  { id: '9:16', w: 9, h: 16, zh: '手机竖屏', en: 'Mobile portrait' },
  { id: '5:4', w: 5, h: 4, zh: '横幅', en: 'Landscape' },
  { id: '4:3', w: 4, h: 3, zh: '经典横幅', en: 'Classic landscape' },
  { id: '3:2', w: 3, h: 2, zh: '摄影', en: 'Photo' },
  { id: '16:9', w: 16, h: 9, zh: '宽屏', en: 'Widescreen' },
  { id: '21:9', w: 21, h: 9, zh: '超宽屏', en: 'Ultrawide' },
]

const MP_OPTIONS = [0.5, 0.6, 0.75, 0.8, 1, 1.25, 1.5, 1.75, 2]

const initialConfig = {
  dataset: {
    inputPath: '',
    repeats: 1,
    ratio: '1:1',
    megapixels: 1,
    enableArBucket: true,
    minAr: 0.5,
    maxAr: 2,
    arBuckets: 7,
    validationEnabled: false,
    validationSubdir: 'validation',
    maskPath: '',
    skipEmptyCaption: true,
  },
  model: {
    architecture: 'Anima',
    paths: {},
    dtype: 'float16',
    transformerDtype: 'bfloat16',
    timestepSampleMethod: 'logit_normal',
  },
  adapter: {
    type: 'LoRA',
    rank: 32,
    decomposeFactor: -1,
    dropout: 0.1,
    learningRate: '1e-4',
    unetLearningRate: '1e-4',
    textEncoderLearningRate: '5e-5',
    textEncoder2LearningRate: '5e-5',
    initFrom: '',
  },
  training: {
    outputFolderName: 'anima_lora_training',
    outputPath: '',
    epochs: 20,
    microBatchSizePerGpu: 2,
    gradientAccumulationSteps: 4,
    warmupSteps: 100,
    checkpointInterval: 1000,
    logInterval: 1,
    optimizer: 'AdamW',
    lrScheduler: 'Cosine',
    saveDtype: 'bfloat16',
    activationCheckpointing: true,
    blocksToSwap: 20,
  },
  startArgs: {
    resumeFromCheckpoint: '',
    regenerateCache: false,
    trustCache: false,
    cacheOnly: false,
    resetDataloader: false,
    resetOptimizerParams: false,
  },
  resources: {
    gpuCount: 1,
    pipelineStages: 1,
    cachingBatchSize: 1,
    partitionMethod: 'parameters',
    mixedPrecision: true,
  },
}

const navItems = [
  { id: 'home', icon: Home, zh: '主页', en: 'Home' },
  { id: 'dataset', icon: FolderOpen, zh: '数据集设置', en: 'Dataset' },
  { id: 'tagger', icon: Tags, zh: 'WD14 自动打标', en: 'WD14 Tagger' },
  { id: 'training', icon: Settings2, zh: '训练设置', en: 'Training' },
  { id: 'start', icon: Play, zh: '启动训练', en: 'Run' },
  { id: 'resources', icon: SlidersHorizontal, zh: '资源调度', en: 'Resources' },
  { id: 'diagnostics', icon: Activity, zh: '环境诊断', en: 'Diagnostics' },
  { id: 'license', icon: BadgeCheck, zh: '许可证', en: 'License' },
]

function pick(lang, zh, en) {
  return lang === 'zh' ? zh : en
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  const total = Math.round(seconds)
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}` : `${minutes}:${String(secs).padStart(2, '0')}`
}

function cleanGreeting(value) {
  return String(value || '').replace(/[。.!！?？；;：:]+$/u, '')
}

function align64(value) {
  return Math.max(64, Math.round(value / 64) * 64)
}

function calculateResolution(ratioId, megapixels) {
  const ratio = RATIOS.find((item) => item.id === ratioId) || RATIOS[0]
  const area = Number(megapixels) * 1_000_000
  const rawWidth = Math.sqrt(area * (ratio.w / ratio.h))
  const rawHeight = Math.sqrt(area * (ratio.h / ratio.w))
  const width = align64(rawWidth)
  const height = align64(rawHeight)
  return {
    width,
    height,
    actualMp: Number(((width * height) / 1_000_000).toFixed(2)),
  }
}

function IconWell({ children, size = 'normal', className = '' }) {
  return <span className={`icon-well icon-well--${size} ${className}`}>{children}</span>
}

function GlassPanel({ children, className = '', glow = false }) {
  return <section className={`glass-panel ${glow ? 'glass-panel--glow' : ''} ${className}`}>{children}</section>
}

function SectionHeading({ icon: Icon, title, subtitle, action }) {
  return (
    <div className="section-heading">
      <div className="section-heading__lead">
        <IconWell size="small"><Icon size={17} /></IconWell>
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  )
}

function Field({ label, hint, children, className = '' }) {
  return (
    <label className={`field ${className}`}>
      <span className="field__label">{label}</span>
      {children}
      {hint && <span className="field__hint">{hint}</span>}
    </label>
  )
}

function SelectField({ label, value, onChange, options, className = '', disabled = false }) {
  return (
    <Field label={label} className={className}>
      <span className="select-shell">
        <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
          {options.map((option) => {
            const optionValue = typeof option === 'string' ? option : option.value
            const optionLabel = typeof option === 'string' ? option : option.label
            return <option key={optionValue} value={optionValue}>{optionLabel}</option>
          })}
        </select>
        <ChevronDown size={16} />
      </span>
    </Field>
  )
}

function InputField({ label, value, onChange, type = 'text', min, max, step, placeholder, className = '', disabled = false, hint = '' }) {
  const inputRef = useRef(null)
  const adjust = (direction) => {
    if (direction > 0) inputRef.current.stepUp()
    else inputRef.current.stepDown()
    onChange(Number(inputRef.current.value))
  }
  return (
    <Field label={label} className={className} hint={hint}>
      <span className={type === 'number' ? 'number-shell' : 'text-shell'}>
      <input
        ref={inputRef}
        type={type}
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(event) => onChange(type === 'number' ? Number(event.target.value) : event.target.value)}
      />
      {type === 'number' && <span className="number-controls">
        <button type="button" aria-label={`${label} +`} disabled={disabled || (max !== undefined && Number(value) >= max)} onClick={(event) => { event.preventDefault(); adjust(1) }}><ChevronDown className="step-up" size={12} /></button>
        <button type="button" aria-label={`${label} −`} disabled={disabled || (min !== undefined && Number(value) <= min)} onClick={(event) => { event.preventDefault(); adjust(-1) }}><ChevronDown size={12} /></button>
      </span>}
      </span>
    </Field>
  )
}

function PathField({ label, value, onChange, onBrowse, placeholder, kind = 'folder', className = '' }) {
  const BrowseIcon = kind === 'folder' ? FolderOpen : FileCode2
  return (
    <Field label={label} className={className}>
      <span className="path-shell">
        <input value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} />
        <button type="button" className="field-icon-button" onClick={onBrowse} aria-label="browse">
          <BrowseIcon size={18} />
        </button>
      </span>
    </Field>
  )
}

function Toggle({ checked, onChange, label, description, compact = false }) {
  return (
    <button type="button" className={`toggle-row ${compact ? 'toggle-row--compact' : ''}`} onClick={() => onChange(!checked)}>
      {(label || description) && (
        <span className="toggle-row__copy">
          {label && <strong>{label}</strong>}
          {description && <small>{description}</small>}
        </span>
      )}
      <span className={`switch ${checked ? 'is-on' : ''}`} aria-hidden="true"><span /></span>
    </button>
  )
}

function ActionButton({ children, icon: Icon, onClick, variant = 'secondary', disabled = false, type = 'button' }) {
  return (
    <button type={type} className={`action-button action-button--${variant}`} onClick={onClick} disabled={disabled}>
      {Icon && <Icon size={17} />}
      <span>{children}</span>
    </button>
  )
}

function StatPill({ icon: Icon, label, value, tone = 'blue' }) {
  return (
    <div className={`stat-pill stat-pill--${tone}`}>
      <IconWell size="small"><Icon size={16} /></IconWell>
      <span><small>{label}</small><strong>{value}</strong></span>
    </div>
  )
}

function PageHeader({ eyebrow, title, description, trailing }) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <span className="page-header__eyebrow"><Sparkles size={14} /> {eyebrow}</span>}
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {trailing}
    </header>
  )
}

function HomePage({ lang, setPage, config, resolution, humanized = true }) {
  const [hour, setHour] = useState(() => new Date().getHours())
  const [phrase] = useState(() => Math.floor(Math.random() * 5))
  useEffect(() => {
    const refresh = () => setHour(new Date().getHours())
    const timer = setInterval(refresh, 30000)
    window.addEventListener('focus', refresh)
    return () => { clearInterval(timer); window.removeEventListener('focus', refresh) }
  }, [])
  const model = <strong className="architecture-highlight">{config.model.architecture}</strong>
  const greetings = hour < 5 ? ['凌晨了，用 ', 'A new day begins with ']
    : hour < 9 ? ['早上好，用 ', 'Good morning. Start a new day with ']
    : hour < 12 ? ['上午好，今天想让 ', 'Good morning. What will you teach ']
    : hour < 14 ? ['中午好，为 ', 'Good afternoon. Make room for ']
    : hour < 18 ? ['下午好，让灵感在 ', 'Good afternoon. Bring your ideas to life with ']
    : hour < 22 ? ['晚上好，和 ', 'Good evening. Shape your next idea with ']
    : ['深夜了，保存好 ', 'It’s late. Save your ']
  const endings = hour < 5 ? ' 的训练开始新的一天' : hour < 9 ? ' 的训练开启清晨的创作' : hour < 12 ? ' 学会什么' : hour < 14 ? ' 留一点新的灵感' : hour < 18 ? ' 的训练中成形' : hour < 22 ? ' 一起完善今天的创作' : ' 的方案，就去休息吧'
  const phrases = lang === 'zh'
    ? [['为 ', ' 准备好你的训练方案'], ['下一份灵感，交给 ', ''], ['', ' 的新旅程，从这组参数开始'], ['让 ', ' 学会你的创作风格'], ['你的创意，', ' 的下一课']]
    : [['Your ', ' training recipe'], ['Your next idea starts with ', ''], ['A fresh chapter for ', ''], ['Teach ', ' your creative style'], ['A new lesson for ', '']]
  const changelog = [
    { version: '0.6.0', date: '2026.09', zh: '接入真实逐步指标、原始日志归档、CLIP Token 预检与核心能力表。', en: 'Added real per-step metrics, raw-log archives, CLIP token preflight, and core capability profiles.' },
    { version: '0.5.2', date: '2026.09', zh: '完善训练桥接、WD14 自动打标与环境诊断。', en: 'Improved the training bridge, WD14 tagging, and environment diagnostics.' },
    { version: '0.5.1', date: '2026.09', zh: '补齐浅色与深色液态玻璃界面。', en: 'Completed the light and dark liquid-glass themes.' },
  ]

  return (
    <div className="page page--home">
      <PageHeader
        eyebrow={pick(lang, '训练工作台', 'Training workspace')}
        title={<>{cleanGreeting(greetings[lang === 'zh' ? 0 : 1])}{model}{humanized ? cleanGreeting(lang === 'zh' ? endings : hour >= 22 ? ' recipe and get some rest.' : ' is ready for your next idea.') : cleanGreeting(lang === 'zh' ? ' 已准备就绪' : ' is ready')}</>}
        description={pick(lang, '从数据集到训练启动，所有关键参数都集中在一个清晰的工作流里。', 'Dataset, model, and runtime controls in one focused workflow.')}
      />

      <GlassPanel className="hero-card" glow>
        <div className="hero-card__copy">
          <span className="status-chip"><span className="status-dot" /> {pick(lang, '训练核心待命', 'Training core ready')}</span>
          <h2>{phrases[phrase][0]}{model}{phrases[phrase][1]}</h2>
          <p>{pick(lang, `当前为 ${resolution.width} × ${resolution.height}、${config.dataset.megapixels} MP、${config.adapter.rank} Rank。你仍可在开始前修改每一项。`, `Current preset: ${resolution.width} × ${resolution.height}, ${config.dataset.megapixels} MP, Rank ${config.adapter.rank}. Everything remains editable.`)}</p>
          <div className="hero-card__actions">
            <ActionButton icon={Play} variant="primary" onClick={() => setPage('start')}>{pick(lang, '检查并启动', 'Review & run')}</ActionButton>
            <ActionButton icon={Settings2} onClick={() => setPage('training')}>{pick(lang, '训练设置', 'Training settings')}</ActionButton>
          </div>
        </div>
        <div className="hero-orbit" aria-hidden="true">
          <div className="hero-orbit__ring hero-orbit__ring--one" />
          <div className="hero-orbit__ring hero-orbit__ring--two" />
          <div className="hero-orbit__core"><WandSparkles size={34} /></div>
          <span className="hero-orbit__tag">{config.model.architecture}</span>
        </div>
      </GlassPanel>

      <div className="home-grid">
        <GlassPanel className="quick-card">
          <SectionHeading icon={Gauge} title={pick(lang, '当前方案', 'Current recipe')} subtitle={pick(lang, '训练前快速确认', 'At-a-glance preflight')} />
          <div className="stat-grid">
            <StatPill icon={Box} label={pick(lang, '架构', 'Architecture')} value={config.model.architecture} tone="violet" />
            <StatPill icon={FileImage} label={pick(lang, '分辨率', 'Resolution')} value={`${resolution.width} × ${resolution.height}`} />
            <StatPill icon={Layers3} label={pick(lang, '适配器', 'Adapter')} value={`${config.adapter.type} · ${config.adapter.rank}`} tone="cyan" />
            <StatPill icon={Clock3} label={pick(lang, '训练轮次', 'Epochs')} value={`${config.training.epochs}`} tone="indigo" />
          </div>
          <button className="inline-link" onClick={() => setPage('dataset')}>{pick(lang, '继续配置数据集', 'Continue dataset setup')} <ArrowRight size={15} /></button>
        </GlassPanel>

        <GlassPanel className="profile-card">
          <div className="profile-card__portrait"><img src="./profile.jpg" alt="profile" /></div>
          <div>
            <span className="micro-label">CREATOR LINK</span>
            <h3>{pick(lang, '闲鱼主页', 'Goofish profile')}</h3>
            <p>{pick(lang, '左下角入口会直接调用系统默认浏览器打开。', 'The sidebar shortcut opens in your system browser.')}</p>
          </div>
          <ExternalLink size={18} />
        </GlassPanel>
      </div>

      <GlassPanel className="changelog-card">
        <SectionHeading icon={BookOpenText} title={pick(lang, '更新日志', 'Changelog')} subtitle={pick(lang, 'Windows 桌面版发布记录', 'Windows desktop release notes')} action={<span className="soft-badge">Stable</span>} />
        <div className="timeline">
          {changelog.map((item, index) => (
            <article className="timeline__item" key={item.version}>
              <span className={`timeline__dot ${index === 0 ? 'is-current' : ''}`} />
              <div className="timeline__meta"><strong>v{item.version}</strong><span>{item.date}</span></div>
              <p>{pick(lang, item.zh, item.en)}</p>
            </article>
          ))}
        </div>
      </GlassPanel>
    </div>
  )
}

function DatasetScanSummary({ lang, datasetStats, scanning }) {
  return (
    <div className={`dataset-scan-summary ${scanning ? 'is-scanning' : ''}`}>
      <IconWell><FileImage size={19} /></IconWell>
      <span>{scanning ? pick(lang, '正在扫描数据集…', 'Scanning dataset…') : pick(lang, '找到图片与标记组合', 'Image and caption pairs found')}</span>
      <strong>{scanning ? '—' : datasetStats.pairs}</strong>
      <small>{pick(lang, '组', 'pairs')}</small>
    </div>
  )
}

function DatasetPage({ lang, config, updateSection, browseFolder, resolution, datasetStats, datasetScanning }) {
  const dataset = config.dataset
  const update = (key, value) => updateSection('dataset', key, value)
  const selectedRatio = RATIOS.find((ratio) => ratio.id === dataset.ratio) || RATIOS[0]

  return (
    <div className="page">
      <PageHeader
        eyebrow={pick(lang, '步骤 1 / 3', 'Step 1 of 3')}
        title={pick(lang, '数据集设置', 'Dataset settings')}
        description={pick(lang, '选择素材目录，定义画幅与像素预算；验证集作为同一数据集中的可选子目录。', 'Choose source media, aspect ratio, and pixel budget. Validation lives here as an optional subfolder.')}
      />

      <GlassPanel>
        <SectionHeading icon={FolderOpen} title={pick(lang, '数据源', 'Data source')} subtitle={pick(lang, '选择图片及其同名标记所在目录', 'Choose the folder containing images and matching captions')} />
        <PathField
          label={pick(lang, '训练数据集根目录', 'Training dataset root')}
          value={dataset.inputPath}
          onChange={(value) => update('inputPath', value)}
          onBrowse={() => browseFolder('dataset')}
          placeholder={pick(lang, '选择包含图片与同名标注的目录', 'Choose a folder containing media and matching captions')}
        />
        <div className="dataset-source-options dataset-source-options--summary">
          <DatasetScanSummary lang={lang} datasetStats={datasetStats} scanning={datasetScanning} />
        </div>
      </GlassPanel>

      <GlassPanel glow>
        <SectionHeading icon={FileImage} title={pick(lang, '智能训练分辨率', 'Smart training resolution')} subtitle={pick(lang, '比例与百万像素共同决定最终宽 × 高，结果自动对齐到 64。', 'Aspect ratio and megapixels determine width × height, aligned to 64.')} />
        <div className="resolution-layout">
          <div className="resolution-step">
            <span className="step-number">1</span>
            <div className="resolution-step__header"><strong>{pick(lang, '选择比例', 'Aspect ratio')}</strong><span>{selectedRatio.id} · {pick(lang, selectedRatio.zh, selectedRatio.en)}</span></div>
            <div className="ratio-grid">
              {RATIOS.map((ratio) => (
                <button key={ratio.id} type="button" className={dataset.ratio === ratio.id ? 'is-selected' : ''} onClick={() => update('ratio', ratio.id)}>
                  <span className="ratio-shape" style={{ '--ratio-w': ratio.w, '--ratio-h': ratio.h }} />
                  <strong>{ratio.id}</strong>
                  <small>{pick(lang, ratio.zh, ratio.en)}</small>
                </button>
              ))}
            </div>
          </div>
          <div className="resolution-step">
            <span className="step-number">2</span>
            <div className="resolution-step__header"><strong>{pick(lang, '百万像素', 'Megapixels')}</strong><span>{dataset.megapixels} MP</span></div>
            <div className="mp-control">
              <input type="range" min="0.5" max="2" step="0.05" value={dataset.megapixels} onChange={(event) => update('megapixels', Number(event.target.value))} />
              <div className="mp-scale"><span>0.5</span><span>1.0</span><span>1.5</span><span>2.0 MP</span></div>
              <div className="mp-chips">
                {MP_OPTIONS.map((mp) => <button key={mp} type="button" className={dataset.megapixels === mp ? 'is-selected' : ''} onClick={() => update('megapixels', mp)}>{mp}</button>)}
              </div>
            </div>
          </div>
          <div className="resolution-result">
            <span className="step-number step-number--result">3</span>
            <span className="resolution-result__eyebrow">{pick(lang, '确定分辨率', 'Final resolution')}</span>
            <div className="resolution-result__value"><strong>{resolution.width}</strong><span>×</span><strong>{resolution.height}</strong></div>
            <div className="resolution-result__canvas"><span style={{ aspectRatio: `${resolution.width} / ${resolution.height}` }}><Sparkles size={22} /></span></div>
            <p>{selectedRatio.id} · {resolution.actualMp} MP · {pick(lang, '64 像素对齐', '64px aligned')}</p>
          </div>
        </div>
      </GlassPanel>

      <div className="two-column">
        <GlassPanel>
          <SectionHeading icon={SlidersHorizontal} title={pick(lang, '分桶与重复', 'Buckets & repeats')} />
          <div className="form-grid form-grid--three">
            <InputField label={pick(lang, '重复次数', 'Repeats')} value={dataset.repeats} onChange={(value) => update('repeats', value)} type="number" min={1} />
            <InputField label={pick(lang, '最小比例', 'Min ratio')} value={dataset.minAr} onChange={(value) => update('minAr', value)} type="number" min={0.1} step={0.1} />
            <InputField label={pick(lang, '最大比例', 'Max ratio')} value={dataset.maxAr} onChange={(value) => update('maxAr', value)} type="number" min={0.1} step={0.1} />
            <InputField label={pick(lang, '比例桶数量', 'AR buckets')} value={dataset.arBuckets} onChange={(value) => update('arBuckets', value)} type="number" min={1} />
            <div className="form-span-2"><Toggle checked={dataset.enableArBucket} onChange={(value) => update('enableArBucket', value)} label={pick(lang, '启用宽高比分桶', 'Enable aspect buckets')} description={pick(lang, '减少裁切并保留构图', 'Preserve composition with less cropping')} /></div>
          </div>
        </GlassPanel>

        <GlassPanel className={dataset.validationEnabled ? 'validation-panel is-enabled' : 'validation-panel'}>
          <SectionHeading
            icon={BadgeCheck}
            title={pick(lang, '验证子目录（可选）', 'Validation subfolder (optional)')}
            action={<Toggle compact checked={dataset.validationEnabled} onChange={(value) => update('validationEnabled', value)} />}
          />
          <p className="panel-copy">{pick(lang, '在训练根目录下放置独立验证素材，不需要再去另一个页面设置。', 'Keep independent validation media under the training root—no separate page required.')}</p>
          <InputField label={pick(lang, '子目录名称或相对路径', 'Subfolder name or relative path')} value={dataset.validationSubdir} onChange={(value) => update('validationSubdir', value)} placeholder="validation" />
          <div className="resolved-path"><Folder size={16} /><span>{dataset.inputPath ? `${dataset.inputPath}\\${dataset.validationSubdir}` : `…\\dataset\\${dataset.validationSubdir}`}</span></div>
        </GlassPanel>
      </div>

      <GlassPanel>
        <SectionHeading icon={FileCode2} title={pick(lang, '标注与遮罩', 'Captions & masks')} />
        <div className="form-grid form-grid--two">
          <PathField label={pick(lang, '遮罩目录（可选）', 'Mask folder (optional)')} value={dataset.maskPath} onChange={(value) => update('maskPath', value)} onBrowse={() => browseFolder('mask')} placeholder={pick(lang, '留空则不使用遮罩', 'Leave empty to train without masks')} />
          <Toggle checked={dataset.skipEmptyCaption} onChange={(value) => update('skipEmptyCaption', value)} label={pick(lang, '跳过无标注素材', 'Skip media without captions')} description={pick(lang, '避免空提示词样本进入训练', 'Keep empty-caption samples out of training')} />
        </div>
      </GlassPanel>
    </div>
  )
}

function WD14Page({ lang, config, setConfig, runtime, logs, scan, diagnostics, browseFolder, startTagger, stopTagger }) {
  const update = (key, value) => setConfig((current) => ({ ...current, [key]: value }))
  const progress = runtime.total ? Math.round((runtime.current / runtime.total) * 100) : 0
  return (
    <div className="page page--tagger">
      <PageHeader eyebrow={pick(lang, '数据准备工具', 'Dataset preparation')} title={pick(lang, 'WD14 自动打标', 'WD14 Auto Tagger')} description={pick(lang, '递归扫描全部子目录，为每张图片生成同名 TXT，并在写入前处理触发词与屏蔽词。', 'Recursively scan subfolders, create matching TXT captions, and apply trigger/block lists before writing.')} trailing={<span className={`run-status run-status--${runtime.status}`}><span />{runtime.label}</span>} />
      <div className="tagger-grid">
        <div>
          <GlassPanel glow>
            <SectionHeading icon={Tags} title="WD EVA02 Large Tagger v3" subtitle={pick(lang, 'SmilingWolf 官方 v3 大模型 · 首次运行自动下载', 'SmilingWolf v3 large model · downloads on first run')} action={<span className="soft-badge">ONNX</span>} />
            <PathField label={pick(lang, '图片根目录', 'Image root')} value={config.rootPath} onChange={(value) => update('rootPath', value)} onBrowse={browseFolder} placeholder={pick(lang, '选择图片目录，将自动递归子文件夹', 'Choose an image folder; subfolders are included')} />
            <div className="tagger-scan"><FolderOpen size={18} /><span>{pick(lang, '递归扫描到', 'Recursive scan')}</span><strong>{scan.scanning ? '…' : scan.total}</strong><small>{pick(lang, '张图片', 'images')}</small></div>
            <div className={`tagger-provider ${diagnostics?.cudaAdvertised ? 'is-ready' : 'is-cpu'}`}><Cpu size={16}/><span>{diagnostics ? (diagnostics.cudaAdvertised ? pick(lang, 'ONNX CUDA Provider 已发现；开始时将实测加载。', 'ONNX CUDA Provider found; model loading will verify it.') : pick(lang, '未发现 ONNX CUDA Provider：将使用 CPU，速度较慢。', 'No ONNX CUDA Provider: CPU will be used and is slower.')) : pick(lang, '正在检测 ONNX Runtime…', 'Checking ONNX Runtime…')}</span></div>
          </GlassPanel>
          <GlassPanel>
            <SectionHeading icon={WandSparkles} title={pick(lang, '标签处理', 'Tag processing')} />
            <div className="form-grid form-grid--two">
              <InputField label={pick(lang, '额外触发词（逗号分隔）', 'Trigger words (comma-separated)')} value={config.triggerWords} onChange={(value) => update('triggerWords', value)} placeholder="my_character, my_style" />
              <InputField label={pick(lang, '屏蔽词（逗号分隔）', 'Blocked words (comma-separated)')} value={config.blockedWords} onChange={(value) => update('blockedWords', value)} placeholder="text, watermark" />
              <InputField label={pick(lang, '通用标签阈值', 'General threshold')} value={config.threshold} onChange={(value) => update('threshold', value)} type="number" min={0} max={1} step={0.05} />
              <InputField label={pick(lang, '角色标签阈值', 'Character threshold')} value={config.characterThreshold} onChange={(value) => update('characterThreshold', value)} type="number" min={0} max={1} step={0.05} />
              <Toggle checked={config.replaceUnderscores} onChange={(value) => update('replaceUnderscores', value)} label={pick(lang, '下划线转为空格', 'Replace underscores')} description={pick(lang, '使标签更适合作为自然语言标注', 'Make tags easier to read as captions')} />
              <Toggle checked={config.overwrite} onChange={(value) => update('overwrite', value)} label={pick(lang, '覆盖现有 TXT', 'Overwrite existing TXT')} description={pick(lang, '关闭时会跳过已有标注的图片', 'When off, images with captions are skipped')} />
            </div>
          </GlassPanel>
          <GlassPanel className="launch-card">
            <div className="launch-card__icon"><Tags size={26} /></div>
            <div><strong>{runtime.running ? pick(lang, '正在生成标注', 'Tagging images') : pick(lang, '打标器已就绪', 'Tagger ready')}</strong><p>{runtime.message}</p></div>
            {runtime.running ? <ActionButton icon={CircleStop} variant="danger" onClick={stopTagger}>{pick(lang, '停止', 'Stop')}</ActionButton> : <ActionButton icon={Play} variant="primary" onClick={startTagger} disabled={!config.rootPath}>{pick(lang, '开始打标', 'Start tagging')}</ActionButton>}
          </GlassPanel>
        </div>
        <GlassPanel className="console-panel tagger-console">
          <div className="tagger-progress">
            <div><span>{pick(lang, '处理进度', 'Progress')}</span><strong>{runtime.current} / {runtime.total || scan.total}</strong></div>
            <div className="training-progress__track"><span style={{ width: `${progress}%` }} /></div>
            <small>{progress}% · {pick(lang, `已写入 ${runtime.written}，跳过 ${runtime.skipped}，失败 ${runtime.failed}`, `${runtime.written} written, ${runtime.skipped} skipped, ${runtime.failed} failed`)}</small>
          </div>
          <div className="console" role="log">{logs.length ? logs.map((log, index) => <div className={`console__line console__line--${log.stream || 'stdout'}`} key={index}><span>{String(index + 1).padStart(3, '0')}</span><code>{log.line}</code></div>) : <div className="console__empty"><Tags size={28} /><span>{pick(lang, '运行记录会显示在这里。', 'Tagger output will appear here.')}</span></div>}</div>
        </GlassPanel>
      </div>
    </div>
  )
}

function TrainingPage({ lang, config, updateSection, browseModelPath, setPage, resetTraining }) {
  const updateModel = (key, value) => updateSection('model', key, value)
  const updateModelPath = (key, value) => updateModel('paths', { ...(config.model.paths || {}), [key]: value })
  const updateAdapter = (key, value) => updateSection('adapter', key, value)
  const updateTraining = (key, value) => updateSection('training', key, value)
  const modelProfile = getModelProfile(config.model.architecture)
  const changeArchitecture = (value) => {
    updateModel('architecture', value)
    if (!getModelProfile(value).supportsBlockSwap) updateTraining('blocksToSwap', 0)
  }

  return (
    <div className="page">
      <PageHeader
        eyebrow={pick(lang, '步骤 2 / 3', 'Step 2 of 3')}
        title={pick(lang, '训练设置', 'Training settings')}
        description={pick(lang, '配置模型、适配器和训练核心。默认值针对 Anima 的 LoRA 微调。', 'Configure model, adapter, and training core. Defaults target an Anima LoRA workflow.')}
        trailing={<span className="model-chip"><span className="model-chip__orb" />{config.model.architecture}</span>}
      />

      <GlassPanel>
        <SectionHeading icon={Box} title={pick(lang, '模型加载配置', 'Model loading')} subtitle={pick(lang, '支持单文件与预训练目录', 'Single-file and pretrained-folder loading')} />
        <div className="form-grid form-grid--three">
          <SelectField label={pick(lang, '模型架构', 'Architecture')} value={config.model.architecture} onChange={changeArchitecture} options={MODEL_ARCHITECTURES} />
          {modelProfile.fields.map((field) => <PathField key={field.key} label={pick(lang, field.zh, field.en)} value={config.model.paths?.[field.key] || ''} onChange={(value) => updateModelPath(field.key, value)} onBrowse={() => browseModelPath(field.key, field.kind)} kind={field.kind} placeholder={field.required ? pick(lang, '必填', 'Required') : pick(lang, '可选', 'Optional')} />)}
          <SelectField label={pick(lang, '基础模型数据类型', 'Base model dtype')} value={config.model.dtype} onChange={(value) => updateModel('dtype', value)} options={['float16', 'bfloat16', 'float32']} />
          <SelectField label={pick(lang, 'Transformer 数据类型', 'Transformer dtype')} value={config.model.transformerDtype} onChange={(value) => updateModel('transformerDtype', value)} options={['bfloat16', 'float16', 'float8_e4m3fn']} />
          <SelectField label={pick(lang, '时间步采样', 'Timestep sampling')} value={modelProfile.supportsTimestepSampling === false ? '核心固定' : config.model.timestepSampleMethod} onChange={(value) => updateModel('timestepSampleMethod', value)} options={modelProfile.supportsTimestepSampling === false ? ['核心固定'] : ['logit_normal', 'uniform']} disabled={modelProfile.supportsTimestepSampling === false} />
        </div>
      </GlassPanel>

      <GlassPanel>
        <SectionHeading icon={Layers3} title={pick(lang, '适配器配置', 'Adapter')} subtitle={pick(lang, 'LoRA / LoKr / 全量微调', 'LoRA, LoKr, or full fine-tuning')} />
        <div className="form-grid form-grid--four compact-fields">
          <SelectField label={pick(lang, '适配器类型', 'Adapter type')} value={config.adapter.type} onChange={(value) => updateAdapter('type', value)} options={['LoRA', 'LoKr', 'None (FFT)']} />
          <InputField label="Rank（秩）" value={config.adapter.rank} onChange={(value) => updateAdapter('rank', value)} type="number" min={1} disabled={config.adapter.type === 'None (FFT)'} />
          {config.adapter.type === 'LoKr' && <InputField label={pick(lang, '分解因子', 'Decompose factor')} value={config.adapter.decomposeFactor} onChange={(value) => updateAdapter('decomposeFactor', value)} type="number" min={-1} />}
          <InputField label={config.adapter.type === 'LoKr' ? 'Rank Dropout' : 'Dropout'} value={config.adapter.dropout} onChange={(value) => updateAdapter('dropout', value)} type="number" min={0} max={1} step={0.01} disabled={config.adapter.type === 'None (FFT)'} />
          <InputField label={pick(lang, '学习率', 'Learning rate')} value={config.adapter.learningRate} onChange={(value) => updateAdapter('learningRate', value)} />
          {modelProfile.lrMode === 'sdxl' && <InputField label={pick(lang, 'UNet 学习率', 'UNet learning rate')} value={config.adapter.unetLearningRate} onChange={(value) => updateAdapter('unetLearningRate', value)} />}
          {modelProfile.lrMode === 'sdxl' && <InputField label={pick(lang, '文本编码器 1 学习率', 'Text encoder 1 LR')} value={config.adapter.textEncoderLearningRate} onChange={(value) => updateAdapter('textEncoderLearningRate', value)} />}
          {modelProfile.lrMode === 'sdxl' && <InputField label={pick(lang, '文本编码器 2 学习率', 'Text encoder 2 LR')} value={config.adapter.textEncoder2LearningRate} onChange={(value) => updateAdapter('textEncoder2LearningRate', value)} />}
          {modelProfile.lrMode === 'anima' && <InputField label={pick(lang, 'LLM Adapter 学习率', 'LLM adapter LR')} value={config.adapter.textEncoderLearningRate} onChange={(value) => updateAdapter('textEncoderLearningRate', value)} />}
          <PathField label={pick(lang, '从现有权重初始化', 'Initialize from weights')} value={config.adapter.initFrom} onChange={(value) => updateAdapter('initFrom', value)} onBrowse={() => browseFile('adapter')} kind="file" placeholder={pick(lang, '可选', 'Optional')} />
        </div>
      </GlassPanel>

      <GlassPanel glow>
        <SectionHeading icon={Cpu} title={pick(lang, '训练核心', 'Training core')} subtitle={pick(lang, '保存、批次和调度策略', 'Saving, batching, and scheduler policy')} />
        <div className="form-grid form-grid--four compact-fields">
          <InputField label={pick(lang, '输出文件夹名称', 'Output folder name')} value={config.training.outputFolderName} onChange={(value) => updateTraining('outputFolderName', value)} />
          <PathField label={pick(lang, '输出根目录', 'Output root')} value={config.training.outputPath} onChange={(value) => updateTraining('outputPath', value)} onBrowse={() => browseModelPath('__output__', 'folder')} kind="folder" placeholder={pick(lang, '留空则使用整合包 output 目录', 'Defaults to the trainer output folder')} />
          <InputField label="Epochs" value={config.training.epochs} onChange={(value) => updateTraining('epochs', value)} type="number" min={1} />
          <InputField label={pick(lang, '每 GPU 批大小', 'Batch size per GPU')} value={config.training.microBatchSizePerGpu} onChange={(value) => updateTraining('microBatchSizePerGpu', value)} type="number" min={1} />
          <InputField label={pick(lang, '梯度累积步数', 'Gradient accumulation')} value={config.training.gradientAccumulationSteps} onChange={(value) => updateTraining('gradientAccumulationSteps', value)} type="number" min={1} />
          <SelectField label={pick(lang, '优化器', 'Optimizer')} value={config.training.optimizer} onChange={(value) => updateTraining('optimizer', value)} options={['AdamW', 'AdamW8bit', 'AdamW8bitKahan', 'Prodigy', 'Lion']} />
          <SelectField label={pick(lang, '学习率调度器', 'LR scheduler')} value={config.training.lrScheduler} onChange={(value) => updateTraining('lrScheduler', value)} options={['Cosine', 'Constant', 'Linear']} />
          <InputField label={pick(lang, 'Warmup 步数', 'Warmup steps')} value={config.training.warmupSteps} onChange={(value) => updateTraining('warmupSteps', value)} type="number" min={0} />
          <InputField label={pick(lang, '模型保存间隔（步）', 'Model save interval')} value={config.training.checkpointInterval} onChange={(value) => updateTraining('checkpointInterval', value)} type="number" min={1} />
          <InputField label={pick(lang, 'DeepSpeed 日志间隔（步）', 'DeepSpeed log interval')} value={config.training.logInterval} onChange={(value) => updateTraining('logInterval', value)} type="number" min={1} hint={pick(lang, '结构化 UI 指标始终逐步采集；建议保留 1', 'Structured UI metrics are captured every step; keep 1 for detailed raw logs')} />
          <SelectField label={pick(lang, '保存精度', 'Save dtype')} value={config.training.saveDtype} onChange={(value) => updateTraining('saveDtype', value)} options={['bfloat16', 'float16', 'float32']} />
          <InputField
            label={pick(lang, '交换到内存的块', 'Blocks to swap')}
            value={modelProfile.supportsBlockSwap ? config.training.blocksToSwap : 0}
            onChange={(value) => updateTraining('blocksToSwap', value)}
            type="number"
            min={0}
            disabled={!modelProfile.supportsBlockSwap}
            hint={!modelProfile.supportsBlockSwap ? pick(lang, '当前架构不支持，已固定为 0', 'Unsupported by this architecture; fixed at 0') : ''}
          />
          <Toggle checked={config.training.activationCheckpointing} onChange={(value) => updateTraining('activationCheckpointing', value)} label={pick(lang, '激活检查点', 'Activation checkpointing')} description={pick(lang, '降低显存占用', 'Reduce VRAM usage')} />
        </div>
        <div className="panel-footer-actions">
          <ActionButton icon={RotateCcw} onClick={resetTraining}>{pick(lang, '恢复默认', 'Restore defaults')}</ActionButton>
          <ActionButton icon={ArrowRight} variant="primary" onClick={() => setPage('start')}>{pick(lang, '保存并继续', 'Save & continue')}</ActionButton>
        </div>
      </GlassPanel>
    </div>
  )
}

function CheckRow({ ok, title, detail }) {
  return (
    <div className={`check-row ${ok ? 'is-ok' : 'is-warning'}`}>
      <span>{ok ? <Check size={16} /> : <Info size={16} />}</span>
      <div><strong>{title}</strong><small>{detail}</small></div>
    </div>
  )
}

function sparkPath(values, width = 240, height = 54) {
  const points = values.filter((value) => Number.isFinite(value))
  if (points.length < 2) return ''
  const min = Math.min(...points)
  const max = Math.max(...points)
  return points.map((value, index) => {
    const x = (index / (points.length - 1)) * width
    const y = height - ((value - min) / Math.max(0.0001, max - min)) * (height - 8) - 4
    return `${index ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

function MiniChart({ label, value, values, unit = '' }) {
  const path = sparkPath(values)
  return <div className="mini-chart"><div><small>{label}</small><strong>{value}{unit}</strong></div><svg viewBox="0 0 240 54" preserveAspectRatio="none" aria-label={label}><path className="mini-chart__grid" d="M0,13 H240 M0,27 H240 M0,41 H240"/><path className="mini-chart__line" d={path || 'M0,30 L240,30'}/></svg></div>
}

function cleanLogLine(line) {
  const cleaned = String(line || '')
    .replace(/\x1B\][^\x07]*(?:\x07|\x1B\\)/g, '')
    .replace(/\x1B\[[0-?]*[ -/]*[@-~]/g, '')
    .replace(/[\u0000\u0008]/g, '')
    .replace(/\r/g, '')
    .replace(/[ \t]+$/g, '')
  if (/lnk1181.*(?:aio|cufile)\.lib/i.test(cleaned)) return `（无影响）DeepSpeed 正在探测可选组件：${cleaned}`
  if (/redirects are currently not supported/i.test(cleaned)) return `（无影响）${cleaned}`
  return cleaned
}

function logKind(line) {
  const value = cleanLogLine(line).replace(/\s+/g, ' ').trim().toLowerCase()
  if (/lnk1181.*(?:aio|cufile)\.lib|redirects are currently not supported|cache-system uses symlinks|need pytorch with cu130|comfy_kitchen does not support stochastic fp8/.test(value)) return 'normal'
  if (/error|traceback|exception|fail/.test(value)) return 'error'
  if (/warning|warn|cuda.*fallback/.test(value)) return 'warning'
  if (/checkpoint|saved|cache.*complete|validation.*complete|training.*complete|finished/.test(value)) return 'event'
  if (/step|epoch|loss|learning.?rate|lr[=:]/.test(value)) return 'progress'
  return 'normal'
}

function TrainingConsole({ lang, config, runtime, metrics, monitorStats, history, events, logs, openOutput, stopTraining, setPage, humanized, logLevel, setLogLevel }) {
  const [logsOpen, setLogsOpen] = useState(false)
  const [confirmStop, setConfirmStop] = useState(false)
  const [range, setRange] = useState('10m')
  const gpu = monitorStats?.gpus?.[0]
  const memory = monitorStats?.memory
  const progress = metrics.totalSteps ? Math.min(100, (metrics.currentStep / metrics.totalSteps) * 100) : runtime.progress
  const filteredLogs = logs.filter((log) => {
    const kind = log.kind === 'noise' ? 'normal' : (log.kind || logKind(log.line))
    return logLevel === 'full' || (logLevel === 'standard' ? kind !== 'normal' : ['error', 'warning', 'event'].includes(kind))
  })
  const visibleLogs = logLevel === 'full' ? filteredLogs : filteredLogs.slice(-240)
  const recentEvents = events.slice(-8).reverse()
  const sampled = range === '1m' ? history.slice(-60) : range === '10m' ? history.slice(-180) : history
  const eta = metrics.remainingSeconds ? new Date(Date.now() + metrics.remainingSeconds * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'
  const latestCheckpoint = [...events].reverse().find((event) => event.type === 'checkpoint')
  const title = runtime.status === 'completed' ? pick(lang, '训练已完成', 'Training complete') : runtime.status === 'error' ? pick(lang, '训练出现错误', 'Training error') : runtime.status === 'stopped' ? pick(lang, '训练已终止', 'Training stopped') : pick(lang, '训练控制台', 'Training console')

  return (
    <div className="page page--console">
      <header className="console-topbar glass-panel">
        <div><span className="console-topbar__eyebrow">{pick(lang, '训练控制台', 'TRAINING CONSOLE')}</span><h1>{title}</h1><p><strong className="architecture-highlight">{config.model.architecture}</strong><span> · {runtime.label}</span><span> · {pick(lang, '已运行', 'Running for')} {formatDuration(metrics.elapsedSeconds)}</span></p></div>
        <div className="console-topbar__actions"><ActionButton icon={FileCode2} onClick={() => desktop?.system?.reveal(metrics.rawLogPath)} disabled={!metrics.rawLogPath}>{pick(lang, '定位原始日志', 'Reveal raw log')}</ActionButton><ActionButton icon={FolderOpen} onClick={openOutput} disabled={!metrics.outputDir}>{pick(lang, '打开输出目录', 'Open output')}</ActionButton><ActionButton icon={ArrowRight} onClick={() => setPage('training')}>{pick(lang, '返回训练设置', 'Back to settings')}</ActionButton></div>
      </header>

      <GlassPanel className="console-progress" glow>
        <div className="console-progress__heading"><span>{pick(lang, '总进度', 'Overall progress')}</span><strong>{progress.toFixed(1)}%</strong><em>{metrics.currentStep} / {metrics.totalSteps || '—'} {pick(lang, '步', 'steps')}</em></div>
        <div className="console-progress__track"><span style={{ width: `${progress}%` }} /></div>
        <div className="console-stat-grid">
          <div><small>{pick(lang, '每步平均', 'Average step')}</small><strong>{formatDuration(metrics.stepSeconds)}</strong></div>
          <div><small>{pick(lang, '已用时间', 'Elapsed')}</small><strong>{formatDuration(metrics.elapsedSeconds)}</strong></div>
          <div><small>{pick(lang, '剩余时间', 'Remaining')}</small><strong>{formatDuration(metrics.remainingSeconds)}</strong></div>
          <div><small>{pick(lang, '预计完成', 'Estimated finish')}</small><strong>{eta}</strong></div>
          <div><small>Epoch</small><strong>{metrics.epoch || '—'}</strong></div>
          <div><small>Loss</small><strong>{Number.isFinite(metrics.loss) ? metrics.loss.toFixed(4) : '—'}</strong></div>
          <div><small>{pick(lang, '学习率', 'Learning rate')}</small><strong>{metrics.learningRate || '—'}</strong></div>
          <div><small>{pick(lang, '最近检查点', 'Latest checkpoint')}</small><strong>{latestCheckpoint?.label || '—'}</strong></div>
        </div>
      </GlassPanel>

      <GlassPanel className="event-axis">
        <SectionHeading icon={Sparkles} title={pick(lang, '训练事件', 'Training events')} subtitle={humanized ? pick(lang, '关键进度、检查点和训练状态会显示在这里', 'Milestones, checkpoints, and training state are surfaced here') : pick(lang, '关键训练事件', 'Key training events')} action={<span className="event-axis__count">{events.length}</span>} />
        <div className="event-axis__list">{recentEvents.length ? recentEvents.map((event) => <button key={event.id} type="button" className={`event-axis__item event-axis__item--${event.type || 'info'}`} onClick={() => setLogsOpen(true)}><time>{event.time}</time><span>{event.icon}</span><strong>{event.label}</strong><small>{event.detail}</small></button>) : <div className="empty-row"><Activity size={20}/>{pick(lang, '等待训练核心发送事件', 'Waiting for training events')}</div>}</div>
      </GlassPanel>

      <div className="console-charts">
        <GlassPanel><div className="chart-toolbar"><SectionHeading icon={Activity} title={pick(lang, '训练数据', 'Training data')} /><div>{['1m','10m','all'].map((item) => <button key={item} className={range === item ? 'is-active' : ''} onClick={() => setRange(item)}>{item === 'all' ? pick(lang, '全部', 'All') : item}</button>)}</div></div><div className="mini-chart-grid"><MiniChart label="Loss" value={Number.isFinite(metrics.loss) ? metrics.loss.toFixed(4) : '—'} values={sampled.map((item) => item.loss)} /><MiniChart label={pick(lang, '学习率', 'Learning rate')} value={metrics.learningRate || '—'} values={sampled.map((item) => Number(item.learningRate))} /><MiniChart label={pick(lang, '每步时长', 'Step time')} value={formatDuration(metrics.stepSeconds)} values={sampled.map((item) => item.stepSeconds)} /><MiniChart label={pick(lang, '训练速度', 'Speed')} value={metrics.stepSeconds ? `${(1 / metrics.stepSeconds).toFixed(2)} step/s` : '—'} values={sampled.map((item) => item.stepSeconds ? 1 / item.stepSeconds : NaN)} /></div></GlassPanel>
        <GlassPanel><SectionHeading icon={Gauge} title={pick(lang, '硬件数据', 'Hardware')} /><div className="mini-chart-grid"><MiniChart label={pick(lang, 'GPU 利用率', 'GPU usage')} value={`${Math.round(gpu?.gpu_util || 0)}%`} values={sampled.map((item) => item.gpuUtil)} unit="" /><MiniChart label={pick(lang, '显存占用', 'VRAM usage')} value={`${Math.round(gpu?.mem_total ? gpu.mem_used / gpu.mem_total * 100 : 0)}%`} values={sampled.map((item) => item.vram)} /><MiniChart label={pick(lang, 'GPU 温度', 'GPU temperature')} value={Number.isFinite(gpu?.temperature) ? `${gpu.temperature}°C` : '—'} values={sampled.map((item) => item.temperature)} /><MiniChart label={pick(lang, '系统内存', 'System RAM')} value={`${Math.round(memory?.percent || 0)}%`} values={sampled.map((item) => item.memory)} /></div></GlassPanel>
      </div>

      <GlassPanel className="console-panel console-panel--controlled">
        <div className="console-panel__head"><SectionHeading icon={TerminalSquare} title={pick(lang, '命令行', 'Command output')} subtitle={logsOpen ? pick(lang, '原始日志已保留，可按级别筛选', 'Raw logs are retained and can be filtered') : pick(lang, '默认收起，仅显示重要信息', 'Collapsed by default; important messages only')} /><div className="console-panel__tools"><select value={logLevel} onChange={(event) => setLogLevel(event.target.value)}><option value="important">{pick(lang, '重要', 'Important')}</option><option value="standard">{pick(lang, '标准', 'Standard')}</option><option value="full">{pick(lang, '完整', 'Full')}</option></select><button type="button" onClick={() => setLogsOpen((value) => !value)}>{logsOpen ? pick(lang, '收起', 'Collapse') : pick(lang, '展开', 'Expand')}</button></div></div>
        {logsOpen && <div className="console" role="log">{visibleLogs.length ? visibleLogs.map((log, index) => <div className={`console__line console__line--${logKind(log.line)}`} key={`${log.timestamp || index}-${index}`}><span>{String(index + 1).padStart(3, '0')}</span><code>{cleanLogLine(log.line)}</code></div>) : <div className="console__empty"><TerminalSquare size={24}/><span>{pick(lang, '此筛选档位暂无可显示的日志', 'No logs at this filter level')}</span></div>}</div>}
      </GlassPanel>

      {runtime.running && <div className="console-controls glass-rail"><span><span className="status-dot"/>{humanized ? pick(lang, '训练正在稳定执行', 'Training is running steadily') : pick(lang, '训练中', 'Running')}</span><div>{confirmStop ? <><small>{pick(lang, '终止后仅保留最近一次检查点，未保存步数可能丢失', 'Termination keeps the latest checkpoint; unsaved steps may be lost')}</small><ActionButton icon={CircleStop} variant="danger" onClick={stopTraining}>{pick(lang, '确认终止', 'Confirm stop')}</ActionButton><ActionButton onClick={() => setConfirmStop(false)}>{pick(lang, '取消', 'Cancel')}</ActionButton></> : <ActionButton icon={CircleStop} variant="danger" onClick={() => setConfirmStop(true)}>{pick(lang, '终止训练', 'Terminate')}</ActionButton>}</div></div>}
    </div>
  )
}

function TrainingPlan({ lang, config, resolution, datasetStats, backendConfig, onBack, onConfirm }) {
  const [plan, setPlan] = useState(null)
  const [loading, setLoading] = useState(Boolean(desktop))
  const modelPath = Object.values(config.model.paths || {}).find(Boolean) || ''
  const modelName = modelPath.split(/[\\/]/).filter(Boolean).pop() || pick(lang, '尚未选择模型', 'No model selected')
  const exposure = (Number(datasetStats.pairs) || 0) * Number(config.dataset.repeats || 1) * Number(config.training.epochs || 1)
  useEffect(() => {
    let alive = true
    if (!desktop?.training?.preview) { setLoading(false); return () => { alive = false } }
    desktop.training.preview(backendConfig).then((result) => { if (alive && result?.success) setPlan(result) }).catch(() => {}).finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [backendConfig])
  const totalSteps = plan?.totalSteps ?? Math.max(1, Math.floor(exposure / Math.max(1, Number(config.training.microBatchSizePerGpu) * Number(config.training.gradientAccumulationSteps) * Number(config.resources.gpuCount))))
  return <div className="plan-overlay" role="dialog" aria-modal="true" aria-label={pick(lang, '训练计划确认', 'Training plan confirmation')}>
    <GlassPanel className="training-plan" glow>
      <span className="page-header__eyebrow"><Sparkles size={14} /> {pick(lang, '启动前确认', 'Pre-run confirmation')}</span>
      <h2>{pick(lang, '确认训练计划', 'Confirm your training plan')}</h2>
      <p>{pick(lang, '以下总步数由当前核心的训练规则预估；真正开始后会以核心发出的计划事件为准。', 'Steps are estimated using the current core rules and are reconciled with the core plan after launch.')}</p>
      <div className="plan-hero"><small>{pick(lang, '预计优化器总步数', 'Estimated optimizer steps')}</small><strong>{loading ? '…' : totalSteps.toLocaleString()}</strong><span>{pick(lang, '步', 'steps')}</span></div>
      <div className="plan-grid">
        <div><small>{pick(lang, '模型', 'Model')}</small><strong title={modelPath}>{modelName}</strong><em>{config.model.architecture}</em></div>
        <div><small>{pick(lang, '训练样本曝光', 'Sample exposures')}</small><strong>{exposure.toLocaleString()}</strong><em>{`${datasetStats.pairs || 0} ${pick(lang, '组图文', 'pairs')} × ${config.dataset.repeats} repeats × ${config.training.epochs} epochs`}</em></div>
        <div><small>{pick(lang, '每 Epoch', 'Per epoch')}</small><strong>{plan?.stepsPerEpoch ?? '—'} {pick(lang, '步', 'steps')}</strong><em>{plan ? `${plan.samplesPerEpoch} ${pick(lang, '次样本曝光', 'sample exposures')}` : pick(lang, '等待核心核算', 'Awaiting core estimate')}</em></div>
        <div><small>{pick(lang, '分辨率 / 有效批次', 'Resolution / effective batch')}</small><strong>{resolution.width} × {resolution.height}</strong><em>{pick(lang, '有效批次', 'Effective batch')} {config.training.microBatchSizePerGpu * config.training.gradientAccumulationSteps * config.resources.gpuCount}</em></div>
        <div><small>{pick(lang, '适配器', 'Adapter')}</small><strong>{config.adapter.type} · Rank {config.adapter.rank}</strong><em>Dropout {config.adapter.dropout}</em></div>
        <div><small>{pick(lang, '优化器 / 调度器', 'Optimizer / scheduler')}</small><strong>{config.training.optimizer} · {config.training.lrScheduler}</strong><em>Warmup {config.training.warmupSteps} {pick(lang, '步', 'steps')}</em></div>
        <div><small>{pick(lang, 'Latent 缓存策略', 'Latent cache policy')}</small><strong>{config.startArgs.regenerateCache ? pick(lang, '本次重建', 'Rebuild now') : config.startArgs.trustCache ? pick(lang, '复用并信任', 'Trust existing') : pick(lang, '校验后复用', 'Validate then reuse')}</strong><em>{pick(lang, '训练核心要求磁盘 latent 缓存', 'The training core requires disk latent caching')}</em></div>
        <div><small>{pick(lang, '模型保存', 'Model saving')}</small><strong>{pick(lang, '每', 'Every')} {config.training.checkpointInterval} {pick(lang, '步', 'steps')}</strong><em>{config.training.saveDtype}</em></div>
      </div>
      <div className="plan-actions"><ActionButton onClick={onBack}>{pick(lang, '返回修改', 'Back to edit')}</ActionButton><ActionButton icon={Play} variant="primary" onClick={onConfirm} disabled={loading && Boolean(desktop)}>{pick(lang, '确认并开始训练', 'Confirm & start')}</ActionButton></div>
    </GlassPanel>
  </div>
}

function StartPage({ lang, config, updateSection, resolution, datasetStats, backendConfig, runtime, metrics, monitorStats, startTraining, stopTraining, openOutput, logs, clearLogs, history, events, humanized, logLevel, setLogLevel, setPage }) {
  const [planOpen, setPlanOpen] = useState(false)
  if (runtime.status === 'starting' || runtime.status === 'running' || ['completed', 'error', 'stopped'].includes(runtime.status)) return <TrainingConsole lang={lang} config={config} runtime={runtime} metrics={metrics} monitorStats={monitorStats} history={history} events={events} logs={logs} openOutput={openOutput} stopTraining={stopTraining} setPage={setPage} humanized={humanized} logLevel={logLevel} setLogLevel={setLogLevel} />
  const modelProfile = getModelProfile(config.model.architecture)
  const missingModelPaths = modelProfile.fields.filter((field) => field.required && !config.model.paths?.[field.key])
  const checks = [
    { ok: Boolean(config.dataset.inputPath), title: pick(lang, '训练数据集', 'Training dataset'), detail: config.dataset.inputPath || pick(lang, '尚未选择目录', 'No folder selected') },
    { ok: missingModelPaths.length === 0, title: pick(lang, '基础模型', 'Base model'), detail: missingModelPaths.length ? pick(lang, `还缺少：${missingModelPaths.map((field) => field.zh).join('、')}`, `Missing: ${missingModelPaths.map((field) => field.en).join(', ')}`) : `${config.model.architecture} · ${modelProfile.fields.filter((field) => field.required).length} ${pick(lang, '项路径已就绪', 'required paths ready')}` },
    { ok: true, title: pick(lang, '输出方案', 'Output recipe'), detail: `${config.training.outputFolderName} · ${config.training.epochs} Epochs` },
    { ok: true, title: pick(lang, '训练分辨率', 'Training resolution'), detail: `${resolution.width} × ${resolution.height} · ${resolution.actualMp} MP` },
  ]
  const ready = checks.every((check) => check.ok)
  const gpu = monitorStats?.gpus?.[0]
  const memory = monitorStats?.memory
  const progress = metrics.totalSteps ? Math.min(100, (metrics.currentStep / metrics.totalSteps) * 100) : runtime.progress
  const cachePath = config.dataset.inputPath ? `${config.dataset.inputPath}\\cache\\${config.model.architecture}` : `…\\dataset\\cache\\${config.model.architecture}`
  const updateStart = (key, value) => updateSection('startArgs', key, value)

  return (
    <div className="page page--run">
      <PageHeader
        eyebrow={pick(lang, '步骤 3 / 3', 'Step 3 of 3')}
        title={pick(lang, '启动训练', 'Run training')}
        description={pick(lang, '最后检查一次关键配置，然后连接原训练后端。', 'Review the essentials, then hand off to the original training backend.')}
        trailing={<span className={`run-status run-status--${runtime.status.toLowerCase()}`}><span />{runtime.label}</span>}
      />
      <div className="run-grid">
        <div className="run-grid__left">
          <GlassPanel>
            <SectionHeading icon={BadgeCheck} title={pick(lang, '配置预检', 'Preflight')} subtitle={ready ? pick(lang, '关键项目已就绪', 'Essential items are ready') : pick(lang, '还有项目需要填写', 'A few items need attention')} />
            <div className="check-list">{checks.map((check) => <CheckRow key={check.title} {...check} />)}</div>
          </GlassPanel>
          <GlassPanel>
            <SectionHeading icon={Settings2} title={pick(lang, '启动参数', 'Launch options')} />
            <div className="launch-summary">
              <div><small>{pick(lang, '架构', 'Architecture')}</small><strong>{config.model.architecture}</strong></div>
              <div><small>{pick(lang, '适配器', 'Adapter')}</small><strong>{config.adapter.type} · Rank {config.adapter.rank}</strong></div>
              <div><small>{pick(lang, '有效批次', 'Effective batch')}</small><strong>{config.training.microBatchSizePerGpu * config.training.gradientAccumulationSteps * config.resources.gpuCount}</strong></div>
              <div><small>{pick(lang, '调度器', 'Scheduler')}</small><strong>{config.training.lrScheduler}</strong></div>
            </div>
          </GlassPanel>
        <GlassPanel className="cache-card">
            <SectionHeading icon={HardDrive} title={pick(lang, 'Latent 磁盘缓存', 'Latent disk cache')} subtitle={pick(lang, '当前训练核心必须使用 latent 磁盘缓存；这里控制复用或重建，避免虚假的“关闭”开关。', 'This core requires disk latent caching; control reuse or rebuild here instead of a nonfunctional off switch.')} action={<span className="status-chip"><span className="status-dot" />{pick(lang, '核心必需', 'Core required')}</span>} />
            <div className="cache-path"><Folder size={15} /><code>{cachePath}</code></div>
            <div className="cache-options">
              <Toggle compact checked={config.startArgs.regenerateCache} onChange={(value) => updateStart('regenerateCache', value)} label={pick(lang, '重新生成缓存', 'Rebuild cache')} />
              <Toggle compact checked={config.startArgs.trustCache} onChange={(value) => updateStart('trustCache', value)} label={pick(lang, '信任已有缓存', 'Trust existing cache')} />
              <Toggle compact checked={config.startArgs.cacheOnly} onChange={(value) => updateStart('cacheOnly', value)} label={pick(lang, '仅生成缓存', 'Cache only')} />
            </div>
          </GlassPanel>
          <GlassPanel className="launch-card" glow>
            <div className="launch-card__icon"><Zap size={28} /></div>
            <div><strong>{runtime.running ? pick(lang, '训练正在进行', 'Training is running') : pick(lang, '准备交给训练核心', 'Ready for the training core')}</strong><p>{runtime.message}</p></div>
            {runtime.running ? (
              <ActionButton icon={CircleStop} variant="danger" onClick={stopTraining}>{pick(lang, '停止训练', 'Stop')}</ActionButton>
            ) : (
              <ActionButton icon={Play} variant="primary" onClick={() => setPlanOpen(true)} disabled={!ready}>{pick(lang, '查看计划并开始', 'Review plan & start')}</ActionButton>
            )}
          </GlassPanel>
        </div>

        {/* The preflight page deliberately has no live console. GPU telemetry and
            command output begin only after the user confirms the plan and starts. */}
        {false && <GlassPanel className="console-panel console-panel--launch">
          <div className="training-dashboard">
            <div className="training-dashboard__head"><span>{pick(lang, '训练进度', 'Training progress')}</span><strong>{metrics.estimated && metrics.totalSteps ? '≈ ' : ''}{metrics.currentStep} / {metrics.totalSteps || '—'}</strong><button type="button" onClick={openOutput} disabled={!metrics.outputDir}><FolderOpen size={15} />{pick(lang, '打开输出目录', 'Open output')}</button></div>
            <div className="training-progress__track"><span style={{ width: `${progress}%` }} /></div>
            <div className="training-time-grid">
              <div><small>{pick(lang, '每步时长', 'Per step')}</small><strong>{formatDuration(metrics.stepSeconds)}</strong></div>
              <div><small>{pick(lang, '总时长', 'Elapsed')}</small><strong>{formatDuration(metrics.elapsedSeconds)}</strong></div>
              <div><small>{pick(lang, '剩余时长', 'Remaining')}</small><strong>{formatDuration(metrics.remainingSeconds)}</strong></div>
              <div><small>{pick(lang, '完成度', 'Complete')}</small><strong>{Math.round(progress)}%</strong></div>
            </div>
            <div className="training-hardware-grid">
              <Meter icon={Gauge} label={pick(lang, 'GPU 利用率', 'GPU usage')} value={gpu?.gpu_util} detail={gpu?.name || pick(lang, '等待检测', 'Waiting for metrics')} />
              <Meter icon={MemoryStick} label={pick(lang, '显存', 'VRAM')} value={gpu?.mem_total ? (gpu.mem_used / gpu.mem_total) * 100 : 0} detail={`${formatBytes(gpu?.mem_used)} / ${formatBytes(gpu?.mem_total)}`} />
              <Meter icon={MemoryStick} label={pick(lang, '系统内存', 'System RAM')} value={memory?.percent} detail={`${formatBytes(memory?.used)} / ${formatBytes(memory?.total)}`} />
              <div className="temperature-card"><Thermometer size={17} /><span><small>{pick(lang, 'GPU 温度', 'GPU temperature')}</small><strong>{Number.isFinite(gpu?.temperature) ? `${gpu.temperature}°C` : '—'}</strong></span></div>
            </div>
          </div>
          <div className="console-panel__head">
            <SectionHeading icon={TerminalSquare} title={pick(lang, '训练日志', 'Training log')} subtitle={runtime.pid ? `PID ${runtime.pid}` : pick(lang, '等待任务', 'Waiting for a task')} />
            <button type="button" onClick={clearLogs}><Trash2 size={16} />{pick(lang, '清空', 'Clear')}</button>
          </div>
          <div className="console" role="log" aria-live="polite">
            {logs.length === 0 ? (
              <div className="console__empty"><TerminalSquare size={28} /><span>{pick(lang, '启动后，后端输出会实时显示在这里。', 'Backend output will stream here after launch.')}</span></div>
            ) : logs.slice(-160).map((log, index) => (
              <div className={`console__line console__line--${log.kind || log.stream || 'stdout'}`} key={`${log.timestamp || index}-${index}`}><span>{String(logs.length - 159 + index).padStart(3, '0')}</span><code>{cleanLogLine(log.line || JSON.stringify(log))}</code></div>
            ))}
          </div>
          <div className="console-panel__footer"><span className="status-dot" />{desktop ? pick(lang, 'Electron IPC 已连接', 'Electron IPC connected') : pick(lang, '浏览器预览模式', 'Browser preview mode')}<span>{logs.length} lines</span></div>
        </GlassPanel>}
      </div>
      {planOpen && <TrainingPlan lang={lang} config={config} resolution={resolution} datasetStats={datasetStats} backendConfig={backendConfig} onBack={() => setPlanOpen(false)} onConfirm={() => { setPlanOpen(false); startTraining() }} />}
    </div>
  )
}

function Meter({ label, value, detail, icon: Icon }) {
  const safeValue = Math.max(0, Math.min(100, Number(value) || 0))
  return (
    <div className="meter">
      <div className="meter__head"><span><Icon size={16} />{label}</span><strong>{safeValue}%</strong></div>
      <div className="meter__track"><span style={{ width: `${safeValue}%` }} /></div>
      <small>{detail}</small>
    </div>
  )
}

function formatBytes(value) {
  if (!value) return '—'
  const gib = value / 1024 / 1024 / 1024
  return `${gib.toFixed(gib >= 10 ? 0 : 1)} GB`
}

function ResourcePage({ lang, config, updateSection, monitorStats, monitorRunning, toggleMonitor }) {
  const update = (key, value) => updateSection('resources', key, value)
  const gpu = monitorStats?.gpus?.[0]
  const memory = monitorStats?.memory
  return (
    <div className="page">
      <PageHeader
        eyebrow={pick(lang, '本次任务', 'Current job')}
        title={pick(lang, '资源调度', 'Resource scheduling')}
        description={pick(lang, '决定训练如何分配到 GPU、系统内存与缓存；这里只显示当前页的即时快照。', 'Choose how the job uses GPU, memory, and caching. Metrics are scoped to this page.')}
        trailing={<ActionButton icon={monitorRunning ? CircleStop : Activity} onClick={toggleMonitor}>{monitorRunning ? pick(lang, '停止读取', 'Stop metrics') : pick(lang, '读取硬件', 'Read hardware')}</ActionButton>}
      />
      <div className="resource-overview">
        <GlassPanel className="hardware-card" glow>
          <div className="hardware-card__icon"><Cpu size={28} /></div>
          <div><small>CPU</small><strong>{monitorStats?.cpu_model || pick(lang, '等待硬件信息', 'Waiting for hardware info')}</strong></div>
          <span>{monitorStats?.cpu_percent ?? 0}%</span>
        </GlassPanel>
        <GlassPanel className="hardware-card">
          <div className="hardware-card__icon"><MemoryStick size={28} /></div>
          <div><small>RAM</small><strong>{formatBytes(memory?.total)}</strong></div>
          <span>{memory?.percent ?? 0}%</span>
        </GlassPanel>
        <GlassPanel className="hardware-card">
          <div className="hardware-card__icon"><Gauge size={28} /></div>
          <div><small>GPU</small><strong>{gpu?.name || pick(lang, '未检测到 NVIDIA GPU', 'No NVIDIA GPU detected')}</strong></div>
          <span>{gpu?.gpu_util ?? 0}%</span>
        </GlassPanel>
      </div>
      <div className="two-column resource-columns">
        <GlassPanel>
          <SectionHeading icon={Gauge} title={pick(lang, '即时用量', 'Live usage')} subtitle={monitorRunning ? pick(lang, '每秒刷新', 'Updated every second') : pick(lang, '按需读取，非全局监控', 'On-demand, not a global monitor')} />
          <div className="meter-stack">
            <Meter icon={Cpu} label="CPU" value={monitorStats?.cpu_percent} detail={monitorStats?.cpu_model || '—'} />
            <Meter icon={MemoryStick} label="RAM" value={memory?.percent} detail={`${formatBytes(memory?.used)} / ${formatBytes(memory?.total)}`} />
            <Meter icon={Gauge} label="GPU" value={gpu?.gpu_util} detail={gpu ? `${formatBytes(gpu.mem_used)} / ${formatBytes(gpu.mem_total)} · ${gpu.temperature}°C` : '—'} />
          </div>
        </GlassPanel>
        <GlassPanel>
          <SectionHeading icon={SlidersHorizontal} title={pick(lang, '分配策略', 'Allocation policy')} />
          <div className="form-grid form-grid--two">
            <InputField label={pick(lang, 'GPU 数量', 'GPU count')} value={config.resources.gpuCount} onChange={(value) => update('gpuCount', value)} type="number" min={1} />
            <InputField label={pick(lang, '管线阶段', 'Pipeline stages')} value={config.resources.pipelineStages} onChange={(value) => update('pipelineStages', value)} type="number" min={1} />
            <InputField label={pick(lang, '缓存批大小', 'Caching batch size')} value={config.resources.cachingBatchSize} onChange={(value) => update('cachingBatchSize', value)} type="number" min={1} />
            <SelectField label={pick(lang, '分区方式', 'Partition method')} value={config.resources.partitionMethod} onChange={(value) => update('partitionMethod', value)} options={['parameters', 'uniform', 'manual']} />
            <div className="form-span-2"><Toggle checked={config.resources.mixedPrecision} onChange={(value) => update('mixedPrecision', value)} label={pick(lang, '混合精度训练', 'Mixed precision')} description={pick(lang, '根据模型精度配置减少显存占用', 'Use model dtypes to reduce VRAM usage')} /></div>
          </div>
        </GlassPanel>
      </div>
      <GlassPanel>
        <SectionHeading icon={HardDrive} title={pick(lang, '磁盘与缓存位置', 'Storage & cache')} />
        <div className="disk-list">
          {(monitorStats?.disks || []).slice(0, 4).map((disk) => (
            <div className="disk-row" key={disk.mountpoint}><HardDrive size={18} /><span><strong>{disk.device || disk.mountpoint}</strong><small>{disk.mountpoint}</small></span><div className="disk-bar"><span style={{ width: `${disk.percent}%` }} /></div><em>{disk.percent}%</em></div>
          ))}
          {!monitorStats?.disks?.length && <div className="empty-row"><HardDrive size={20} />{pick(lang, '读取硬件后显示可用磁盘。', 'Read hardware to show available disks.')}</div>}
        </div>
      </GlassPanel>
    </div>
  )
}

function DiagnosticItem({ icon: Icon, label, value, ok }) {
  return (
    <div className="diagnostic-item">
      <IconWell><Icon size={19} /></IconWell>
      <span><small>{label}</small><strong>{value || '—'}</strong></span>
      <span className={`diagnostic-state ${ok ? 'is-ok' : 'is-error'}`}>{ok ? <Check size={15} /> : <X size={15} />}</span>
    </div>
  )
}

function DiagnosticsPage({ lang, diagnostics, refreshDiagnostics, trainerRoot, setTrainerRoot, browseTrainerRoot }) {
  return (
    <div className="page">
      <PageHeader
        eyebrow={pick(lang, '连接状态', 'Connection status')}
        title={pick(lang, '环境诊断', 'Environment diagnostics')}
        description={pick(lang, '确认 Python、桥接脚本与原整合包目录是否可用。', 'Verify Python, the bridge, and the original all-in-one package root.')}
        trailing={<ActionButton icon={RefreshCw} onClick={refreshDiagnostics}>{pick(lang, '重新检查', 'Run checks')}</ActionButton>}
      />
      <GlassPanel glow>
        <SectionHeading icon={FolderOpen} title={pick(lang, '整合包目录', 'Trainer root')} subtitle={pick(lang, '选择原程序中包含 python、backend、models 等目录的根位置。', 'Choose the original root containing python, backend, models, and related folders.')} />
        <PathField label={pick(lang, '当前目录', 'Current root')} value={trainerRoot} onChange={setTrainerRoot} onBrowse={browseTrainerRoot} placeholder={pick(lang, '自动检测失败时手动选择', 'Choose manually if auto-detection fails')} />
        <div className="callout"><Info size={17} /><span>{pick(lang, '新前端不会移动或重写原训练文件；目录仅用于定位训练核心。', 'The new frontend does not move or rewrite training files; this path only locates the existing backend.')}</span></div>
      </GlassPanel>
      <div className="diagnostic-grid">
        <GlassPanel>
          <SectionHeading icon={Activity} title={pick(lang, '核心检查', 'Core checks')} />
          <div className="diagnostic-list">
            <DiagnosticItem icon={TerminalSquare} label={pick(lang, 'Python 解释器', 'Python interpreter')} value={diagnostics?.pythonVersion || diagnostics?.python} ok={diagnostics?.pythonReady} />
            <DiagnosticItem icon={Database} label={pick(lang, '训练核心', 'Training core')} value={diagnostics?.trainerRoot} ok={diagnostics?.trainerReady} />
            <DiagnosticItem icon={FileCode2} label={pick(lang, '后端桥接', 'Backend bridge')} value={diagnostics?.bridge} ok={diagnostics?.bridgeReady} />
            <DiagnosticItem icon={Activity} label={pick(lang, '硬件读取脚本', 'Hardware metrics script')} value={diagnostics?.monitor} ok={diagnostics?.monitorReady} />
          </div>
        </GlassPanel>
        <GlassPanel>
          <SectionHeading icon={Info} title={pick(lang, '应用信息', 'Application')} />
          <dl className="info-list">
            <div><dt>{pick(lang, '应用版本', 'App version')}</dt><dd>{diagnostics?.appVersion || '0.1.0'}</dd></div>
            <div><dt>{pick(lang, '运行平台', 'Platform')}</dt><dd>{diagnostics?.platform || navigator.platform}</dd></div>
            <div><dt>{pick(lang, '界面引擎', 'UI engine')}</dt><dd>React + Electron</dd></div>
            <div><dt>{pick(lang, '渲染模式', 'Rendering')}</dt><dd>Liquid Glass / GPU</dd></div>
          </dl>
        </GlassPanel>
      </div>
      <GlassPanel>
        <SectionHeading icon={Cloud} title={pick(lang, '检测到的候选目录', 'Detected candidate roots')} />
        <div className="root-list">
          {(diagnostics?.trainerRoots || []).map((root) => <button key={root} type="button" onClick={() => setTrainerRoot(root)}><Folder size={16} /><span>{root}</span><ChevronRight size={16} /></button>)}
          {!diagnostics?.trainerRoots?.length && <div className="empty-row">{pick(lang, '点击“重新检查”获取环境信息。', 'Run checks to inspect the environment.')}</div>}
        </div>
      </GlassPanel>
    </div>
  )
}

function BrandMark() {
  return <img className="brand-mark" src="./profile.jpg" alt="" />
}

function LicensePage({ lang, status, refresh, showToast }) {
  const [busy, setBusy] = useState(false)
  const [code, setCode] = useState('')
  const copyRequest = async () => { await navigator.clipboard?.writeText(status?.requestCode || ''); showToast(pick(lang, '机器请求码已复制。', 'Machine request code copied.')) }
  const importLicense = async () => {
    const selected = await desktop?.dialog?.file({ title: pick(lang, '导入许可证文件', 'Import license file'), filters: [{ name: 'License', extensions: ['lic', 'json'] }] })
    if (!selected) return
    setBusy(true)
    try {
      const result = await desktop.license.activateFile(selected)
      showToast(result.active ? pick(lang, '许可证已激活。', 'License activated.') : `${pick(lang, '激活失败：', 'Activation failed: ')}${result.reason}`)
      refresh()
    } catch (error) { showToast(`${pick(lang, '导入失败：', 'Import failed: ')}${error.message}`) } finally { setBusy(false) }
  }
  return <div className="page"><PageHeader eyebrow={pick(lang, '离线授权', 'Offline licensing')} title={status?.active ? pick(lang, '许可证已激活', 'License active') : pick(lang, '激活许可证', 'Activate license')} description={pick(lang, '许可证在本机离线验证；第一次只需把机器请求码发送给授权方。', 'The license verifies locally; send the request code to the issuer once for first activation.')} />
    <GlassPanel glow className="license-card"><SectionHeading icon={BadgeCheck} title={status?.active ? pick(lang, '授权状态', 'Authorization status') : pick(lang, '机器请求码', 'Machine request code')} />
      {status?.active ? <div className="license-active"><Check size={34}/><div><strong>{status.edition || 'commercial'}</strong><p>{pick(lang, '授权编号：', 'License ID: ')}{status.licenseId}</p><p>{pick(lang, '设备绑定与每次启动的本机密钥挑战均已通过。', 'Device binding and the per-start local-key challenge have passed.')}</p></div></div> : <><p className="license-warning">{status?.reason || pick(lang, '尚未激活。', 'Not activated.')}</p><code className="request-code">{status?.requestCode || '—'}</code><div className="panel-footer-actions"><ActionButton icon={FileCode2} onClick={copyRequest}>{pick(lang, '复制请求码', 'Copy request code')}</ActionButton><ActionButton icon={BadgeCheck} variant="primary" onClick={importLicense} disabled={busy}>{pick(lang, '导入许可证', 'Import license')}</ActionButton></div><textarea className="license-code-input" value={code} onChange={e=>setCode(e.target.value)} placeholder={pick(lang,'粘贴许可证码','Paste license code')} /><ActionButton icon={BadgeCheck} variant="primary" disabled={!code.trim()||busy} onClick={async()=>{setBusy(true);const r=await desktop.license.activate(code);setBusy(false);showToast(r.active?pick(lang,'许可证已激活。','License activated.'):r.reason);refresh()}}>{pick(lang,'验证并激活','Verify & activate')}</ActionButton></>}
    </GlassPanel></div>
}

function LicenseGate({ lang, status, onOpen, onLater }) {
  const [code,setCode]=useState('')
  const [message,setMessage]=useState('')
  const activate=async()=>{const r=await desktop?.license?.activate(code);if(r?.active){onLater();window.location.reload()}else setMessage(r?.reason||pick(lang,'许可证码无效。','Invalid license code.'))}
  const copy=async()=>{await navigator.clipboard?.writeText(status?.requestCode||'');setMessage(pick(lang,'请求码已复制。','Request code copied.'))}
  return <div className="license-gate"><GlassPanel className="license-gate__card" glow><span className="page-header__eyebrow"><BadgeCheck size={14}/> OFFLINE LICENSE</span><h1>{pick(lang,'需要许可证才能使用训练器','A license is required')}</h1><p>{pick(lang,'此设备尚未授权。复制请求码发送给授权方，再粘贴收到的许可证码。','This device is not authorized. Send the request code to the issuer, then paste the license code here.')}</p><label>{pick(lang,'请求码','Request code')}</label><div className="license-gate__request"><code>{status?.requestCode||'—'}</code><button onClick={copy}>{pick(lang,'复制','Copy')}</button></div><label>{pick(lang,'许可证码','License code')}</label><textarea value={code} onChange={e=>setCode(e.target.value)} placeholder={pick(lang,'粘贴许可证码','Paste license code')} /><div className="license-gate__actions"><ActionButton onClick={onLater}>{pick(lang,'稍后再说','Later')}</ActionButton><ActionButton icon={BadgeCheck} variant="primary" disabled={!code.trim()} onClick={activate}>{pick(lang,'验证许可证','Verify license')}</ActionButton></div>{message&&<small>{message}</small>}<button className="license-gate__link" onClick={onOpen}>{pick(lang,'也可稍后在左侧“许可证”中完成激活','You can also activate later from License in the sidebar')}</button></GlassPanel></div>
}

function App() {
  const [page, setPage] = useState('home')
  const [lang, setLang] = useState('zh')
  const [theme, setTheme] = useState('light')
  const [glass, setGlass] = useState(true)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [config, setConfig] = useState(initialConfig)
  const [toast, setToast] = useState('')
  const [logs, setLogs] = useState([])
  const [runtime, setRuntime] = useState({ running: false, status: 'idle', label: '待命', message: '检查配置后即可启动。', progress: 0, pid: null })
  const [trainingMetrics, setTrainingMetrics] = useState({ currentStep: 0, totalSteps: 0, stepSeconds: null, elapsedSeconds: 0, remainingSeconds: null, outputDir: '', rawLogPath: '', estimated: true, startedAt: null, epoch: null, loss: null, learningRate: '' })
  const [trainingHistory, setTrainingHistory] = useState([])
  const [trainingEvents, setTrainingEvents] = useState([])
  const [consoleLogLevel, setConsoleLogLevel] = useState('important')
  const [monitorStats, setMonitorStats] = useState(null)
  const [monitorRunning, setMonitorRunning] = useState(false)
  const [datasetStats, setDatasetStats] = useState({ pairs: 0 })
  const [datasetScanning, setDatasetScanning] = useState(false)
  const [diagnostics, setDiagnostics] = useState(null)
  const [trainerRoot, setTrainerRootState] = useState('')
  const [taggerConfig, setTaggerConfig] = useState({ rootPath: '', triggerWords: '', blockedWords: '', threshold: 0.35, characterThreshold: 0.85, overwrite: false, replaceUnderscores: true })
  const [taggerRuntime, setTaggerRuntime] = useState({ running: false, status: 'idle', label: '待命', message: '选择目录后即可自动打标。', current: 0, total: 0, written: 0, skipped: 0, failed: 0 })
  const [taggerLogs, setTaggerLogs] = useState([])
  const [taggerScan, setTaggerScan] = useState({ total: 0, scanning: false })
  const [license, setLicense] = useState(null)
  const [licenseGateDismissed, setLicenseGateDismissed] = useState(false)
  const [taggerDiagnostics, setTaggerDiagnostics] = useState(null)
  const resolution = useMemo(() => calculateResolution(config.dataset.ratio, config.dataset.megapixels), [config.dataset.ratio, config.dataset.megapixels])

  const t = useCallback((zh, en) => pick(lang, zh, en), [lang])
  const refreshLicense = useCallback(() => desktop?.license?.status?.().then(setLicense).catch(() => {}), [])

  const showToast = useCallback((message) => {
    setToast(message)
    window.clearTimeout(showToast.timer)
    showToast.timer = window.setTimeout(() => setToast(''), 2400)
  }, [])

  const pushTrainingEvent = useCallback((event) => {
    setTrainingEvents((current) => [...current, { id: `${Date.now()}-${Math.random()}`, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), type: 'info', icon: '•', detail: '', ...event }].slice(-120))
  }, [])

  const savePreference = useCallback((patch) => {
    desktop?.preferences?.set(patch).catch(() => {})
  }, [])

  useEffect(() => {
    desktop?.preferences?.get().then((saved) => {
      if (!saved) return
      setLang(saved.language || 'zh')
      setTheme(saved.theme || 'light')
      setGlass(saved.glass !== false)
      setTrainerRootState(saved.trainerRoot || '')
    }).catch(() => {})
  }, [])
  useEffect(() => { refreshLicense() }, [refreshLicense])

  useEffect(() => {
    const removeOutput = desktop?.training?.onOutput((payload) => setLogs((current) => [...current, payload]))
    const removeStatus = desktop?.training?.onStatus((payload) => {
      const status = String(payload?.status || '').toUpperCase()
      if (status === 'RUNNING' || status === 'TRAINING_STARTED') {
        const message = payload.config_path
          ? t(`配置已生成：${payload.config_path}`, `Config generated: ${payload.config_path}`)
          : t('训练核心正在处理任务。', 'The training core is processing the job.')
        setRuntime((current) => ({ ...current, running: true, status: status === 'TRAINING_STARTED' ? 'running' : 'starting', label: status === 'TRAINING_STARTED' ? t('训练中', 'Running') : t('正在启动', 'Starting'), message, pid: payload.pid || current.pid, progress: Math.max(3, current.progress) }))
        if (status === 'TRAINING_STARTED') {
          setTrainingMetrics((current) => ({ ...current, totalSteps: Number(payload.total_steps) || current.totalSteps, outputDir: payload.output_dir || current.outputDir, estimated: payload.total_steps_estimated !== false, startedAt: current.startedAt || Date.now() }))
          pushTrainingEvent({ type: 'start', icon: '▶', label: t('训练已启动', 'Training started'), detail: config.model.architecture })
        }
      } else if (status === 'PROGRESS') {
        setTrainingMetrics((current) => ({ ...current, currentStep: Number(payload.currentStep) || current.currentStep, totalSteps: Number(payload.totalSteps) || current.totalSteps, stepSeconds: Number(payload.stepSeconds) || current.stepSeconds, elapsedSeconds: Number(payload.elapsedSeconds) || current.elapsedSeconds, remainingSeconds: Number.isFinite(Number(payload.remainingSeconds)) ? Number(payload.remainingSeconds) : current.remainingSeconds, outputDir: payload.outputDir || current.outputDir, estimated: payload.estimated !== false, epoch: payload.epoch ?? current.epoch, loss: Number.isFinite(Number(payload.loss)) ? Number(payload.loss) : current.loss, learningRate: payload.learningRate || current.learningRate }))
        setTrainingHistory((current) => [...current, { at: Date.now(), stepSeconds: Number(payload.stepSeconds), loss: Number(payload.loss), learningRate: Number(payload.learningRate) }].slice(-600))
        const checkpoint = Number(payload.currentStep) && Number(payload.checkpointInterval) && Number(payload.currentStep) % Number(payload.checkpointInterval) === 0
        if (checkpoint) pushTrainingEvent({ type: 'checkpoint', icon: '🎉', label: t(`检查点 ${payload.currentStep}`, `Checkpoint ${payload.currentStep}`), detail: t('权重已经保存', 'Weights saved') })
        const percentage = Number(payload.totalSteps) ? Math.floor(Number(payload.currentStep) / Number(payload.totalSteps) * 100) : 0
        if ([10, 25, 50, 75, 90].includes(percentage) && !trainingEvents.some((event) => event.milestone === percentage)) pushTrainingEvent({ type: 'milestone', milestone: percentage, icon: percentage === 50 ? '✨' : '🌱', label: t(`进度达到 ${percentage}%`, `Reached ${percentage}%`), detail: t(percentage === 50 ? '训练已经过半' : '训练正在稳定推进', percentage === 50 ? 'Training is halfway there' : 'Training is progressing steadily') })
      } else if (status === 'TRAINING_PLAN') {
        setTrainingMetrics((current) => ({ ...current, totalSteps: Number(payload.totalSteps) || current.totalSteps, estimated: false }))
      } else if (status === 'RAW_LOG_READY') {
        setTrainingMetrics((current) => ({ ...current, rawLogPath: payload.rawLogPath || '', outputDir: payload.outputDir || current.outputDir }))
      } else if (status === 'CAPTION_SCAN') {
        const overLimit = payload.overLimit || []
        if (!payload.exact) pushTrainingEvent({ type: 'warning', icon: '!', label: t('CLIP Token 扫描未完成', 'CLIP token scan unavailable'), detail: payload.message })
        else if (overLimit.length) {
          pushTrainingEvent({ type: 'warning', icon: '!', label: t(`${overLimit.length} 个标注超过 77 Token`, `${overLimit.length} captions exceed 77 tokens`), detail: overLimit.slice(0, 3).map((item) => `${item.file} (${item.tokenCount ?? '编码错误'})`).join('；') })
          setLogs((current) => [...current, ...overLimit.map((item) => ({ line: `[CLIP 77 Token] ${item.file}: ${item.tokenCount ?? item.error}`, stream: 'stderr', kind: 'warning', timestamp: Date.now() }))])
        } else pushTrainingEvent({ type: 'info', icon: '✓', label: t('CLIP 77 Token 扫描通过', 'CLIP 77-token scan passed'), detail: t(`已精确扫描 ${payload.scanned} 个标注`, `${payload.scanned} captions scanned exactly`) })
      } else if (status === 'CHECKPOINT_SAVED') {
        pushTrainingEvent({ type: 'checkpoint', icon: '✓', label: t('模型已保存', 'Model saved'), detail: payload.message })
      } else if (status === 'WARNING') {
        setLogs((current) => [...current, { line: payload.message || t('训练核心警告', 'Training core warning'), stream: 'stderr', kind: 'warning', timestamp: Date.now() }])
        pushTrainingEvent({ type: 'warning', icon: '!', label: t('训练核心警告', 'Training core warning'), detail: payload.message || '' })
      } else if (['COMPLETED', 'FINISHED'].includes(status)) {
        setRuntime((current) => ({ ...current, running: false, status: 'completed', label: t('已完成', 'Completed'), message: t('训练任务已完成。', 'Training completed.'), progress: 100 }))
        pushTrainingEvent({ type: 'complete', icon: '✓', label: t('训练完成', 'Training complete'), detail: t('最终统计和输出目录仍会保留', 'Final statistics and output remain available') })
      } else if (['ERROR', 'FAILED'].includes(status)) {
        setRuntime((current) => ({ ...current, running: false, status: 'error', label: t('发生错误', 'Error'), message: payload.message || t('训练进程异常结束。', 'The process ended unexpectedly.') }))
        pushTrainingEvent({ type: 'error', icon: '!', label: t('训练出现错误', 'Training error'), detail: payload.message || t('请在命令行中查看首条异常', 'Check the first exception in the command output') })
      } else if (status === 'STOPPED') {
        setRuntime((current) => ({ ...current, running: false, status: 'stopped', label: t('已停止', 'Stopped'), message: t('训练已由用户停止。', 'Training was stopped.') }))
        pushTrainingEvent({ type: 'stop', icon: '■', label: t('训练已终止', 'Training stopped'), detail: t('可从最近检查点继续', 'Resume from the latest checkpoint') })
      }
    })
    const removeStats = desktop?.monitor?.onStats((payload) => {
      if (!payload?.error) {
        setMonitorStats(payload)
        const gpu = payload.gpus?.[0]
        setTrainingHistory((current) => [...current, { at: Date.now(), gpuUtil: Number(gpu?.gpu_util), vram: gpu?.mem_total ? Number(gpu.mem_used) / Number(gpu.mem_total) * 100 : NaN, temperature: Number(gpu?.temperature), memory: Number(payload.memory?.percent) }].slice(-600))
      }
    })
    const removeTaggerOutput = desktop?.tagger?.onOutput((payload) => setTaggerLogs((current) => [...current.slice(-999), payload]))
    const removeTaggerStatus = desktop?.tagger?.onStatus((payload) => {
      const status = String(payload?.status || '').toUpperCase()
      if (['MODEL_PREPARING', 'DOWNLOADING', 'STARTED', 'PROGRESS'].includes(status)) {
        setTaggerRuntime((current) => ({ ...current, running: true, status: 'running', label: status === 'DOWNLOADING' ? t('下载模型中', 'Downloading model') : t('运行中', 'Running'), message: payload.file ? t(`正在处理：${payload.file}${Number.isFinite(payload.tagCount) ? ` · ${payload.tagCount} 标签 · ${Number(payload.seconds || 0).toFixed(2)} 秒` : ''}`, `Processing: ${payload.file}${Number.isFinite(payload.tagCount) ? ` · ${payload.tagCount} tags · ${Number(payload.seconds || 0).toFixed(2)}s` : ''}`) : t('WD14 正在处理图片。', 'WD14 is processing images.'), current: Number(payload.current) || current.current, total: Number(payload.total) || current.total, written: Number(payload.written) || current.written, skipped: Number(payload.skipped) || current.skipped, failed: Number(payload.failed) || current.failed }))
      } else if (status === 'WARNING') {
        setTaggerLogs((current) => [...current, { line: payload.message, stream: 'stderr', kind: 'warning', timestamp: Date.now() }])
      } else if (status === 'COMPLETED') setTaggerRuntime((current) => ({ ...current, running: false, status: 'completed', label: t('已完成', 'Completed'), message: t('全部图片处理完成。', 'All images have been processed.') }))
      else if (['ERROR', 'FAILED'].includes(status)) setTaggerRuntime((current) => ({ ...current, running: false, status: 'error', label: t('发生错误', 'Error'), message: payload.message || t('打标进程异常结束。', 'Tagger process ended unexpectedly.') }))
      else if (status === 'STOPPED') setTaggerRuntime((current) => ({ ...current, running: false, status: 'stopped', label: t('已停止', 'Stopped'), message: t('打标已停止。', 'Tagging stopped.') }))
    })
    return () => {
      removeOutput?.()
      removeStatus?.()
      removeStats?.()
      removeTaggerOutput?.()
      removeTaggerStatus?.()
    }
  }, [config.model.architecture, pushTrainingEvent, t, trainingEvents])

  useEffect(() => {
    if (page !== 'resources' && !runtime.running && monitorRunning) {
      desktop?.monitor?.stop().catch(() => {})
      setMonitorRunning(false)
    }
  }, [page, runtime.running, monitorRunning])

  useEffect(() => {
    if (!runtime.running || !trainingMetrics.startedAt) return
    const timer = window.setInterval(() => setTrainingMetrics((current) => ({ ...current, elapsedSeconds: (Date.now() - current.startedAt) / 1000, remainingSeconds: current.stepSeconds && current.totalSteps ? Math.max(0, current.totalSteps - current.currentStep) * current.stepSeconds : current.remainingSeconds })), 1000)
    return () => window.clearInterval(timer)
  }, [runtime.running, trainingMetrics.startedAt])

  useEffect(() => {
    let active = true
    if (!taggerConfig.rootPath || !desktop?.tagger?.scan) { setTaggerScan({ total: 0, scanning: false }); return () => { active = false } }
    setTaggerScan((current) => ({ ...current, scanning: true }))
    const timer = window.setTimeout(async () => {
      const result = await desktop.tagger.scan(taggerConfig.rootPath).catch(() => ({ success: false, total: 0 }))
      if (active) setTaggerScan({ total: result.success ? result.total : 0, scanning: false })
    }, 180)
    return () => { active = false; window.clearTimeout(timer) }
  }, [taggerConfig.rootPath])

  useEffect(() => {
    desktop?.tagger?.diagnostics?.().then((result) => { if (result?.success) setTaggerDiagnostics(result) }).catch(() => {})
  }, [])

  useEffect(() => {
    let current = true
    const rootPath = config.dataset.inputPath
    if (!rootPath || !desktop?.dataset?.scan) {
      setDatasetStats({ pairs: 0 })
      setDatasetScanning(false)
      return () => { current = false }
    }
    setDatasetScanning(true)
    const timer = window.setTimeout(async () => {
      const result = await desktop.dataset.scan(rootPath, '.txt').catch((error) => ({ success: false, pairs: 0, message: error.message }))
      if (!current) return
      setDatasetStats({ pairs: result.success ? result.pairs : 0 })
      setDatasetScanning(false)
      if (!result.success) showToast(t(`数据集扫描失败：${result.message}`, `Dataset scan failed: ${result.message}`))
    }, 180)
    return () => { current = false; window.clearTimeout(timer) }
  }, [config.dataset.inputPath, showToast, t])

  const updateSection = useCallback((section, key, value) => {
    setConfig((current) => ({ ...current, [section]: { ...current[section], [key]: value } }))
  }, [])

  const resetTraining = useCallback(() => {
    setConfig((current) => ({
      ...current,
      model: { ...initialConfig.model },
      adapter: { ...initialConfig.adapter },
      training: { ...initialConfig.training },
    }))
    showToast(t('训练参数已恢复为 Anima 默认值。', 'Training settings restored to the Anima defaults.'))
  }, [showToast, t])

  const browseFolder = useCallback(async (target) => {
    if (!desktop) {
      showToast(t('请在 Electron 应用中使用系统目录选择器。', 'Use the folder picker inside the Electron app.'))
      return
    }
    const selected = await desktop.dialog.folder({ title: t('选择目录', 'Choose folder') })
    if (!selected) return
    if (target === 'dataset') updateSection('dataset', 'inputPath', selected)
    else if (target === 'mask') updateSection('dataset', 'maskPath', selected)
    else if (target === 'model') updateSection('model', 'modelPath', selected)
    else if (target === 'output') updateSection('training', 'outputPath', selected)
  }, [showToast, t, updateSection])

  const browseFile = useCallback(async (target) => {
    if (!desktop) {
      showToast(t('请在 Electron 应用中使用系统文件选择器。', 'Use the file picker inside the Electron app.'))
      return
    }
    const selected = await desktop.dialog.file({
      title: t('选择权重文件', 'Choose weights'),
      filters: [{ name: 'Model weights', extensions: ['safetensors', 'ckpt', 'pt', 'bin'] }],
    })
    if (!selected) return
    if (target === 'model') updateSection('model', 'modelPath', selected)
    if (target === 'adapter') updateSection('adapter', 'initFrom', selected)
  }, [showToast, t, updateSection])

  const browseModelPath = useCallback(async (key, kind) => {
    if (!desktop) return showToast(t('请在 Electron 应用中使用系统选择器。', 'Use the system picker inside the Electron app.'))
    if (key === '__output__') {
      const selected = await desktop.dialog.folder({ title: t('选择输出根目录', 'Choose output root') })
      if (selected) updateSection('training', 'outputPath', selected)
      return
    }
    const selected = kind === 'folder'
      ? await desktop.dialog.folder({ title: t('选择模型目录', 'Choose model folder') })
      : await desktop.dialog.file({ title: t('选择模型权重', 'Choose model weights'), filters: [{ name: 'Model weights', extensions: ['safetensors', 'ckpt', 'pt', 'bin'] }] })
    if (!selected) return
    setConfig((current) => ({ ...current, model: { ...current.model, paths: { ...(current.model.paths || {}), [key]: selected } } }))
  }, [showToast, t, updateSection])

  const backendConfig = useMemo(() => ({
    dataset: {
      input_path: config.dataset.inputPath,
      resolutions: `[[${resolution.width}, ${resolution.height}]]`,
      enable_ar_bucket: config.dataset.enableArBucket,
      min_ar: config.dataset.minAr,
      max_ar: config.dataset.maxAr,
      num_ar_buckets: config.dataset.arBuckets,
      num_repeats: config.dataset.repeats,
      eval_subdir: config.dataset.validationEnabled ? config.dataset.validationSubdir : '',
      mask_path: config.dataset.maskPath,
      skip_empty_caption: config.dataset.skipEmptyCaption,
    },
    model: {
      architecture: config.model.architecture,
      backend_type: getModelProfile(config.model.architecture).backendType,
      paths: config.model.paths || {},
      defaults: getModelProfile(config.model.architecture).defaults || {},
      supports_block_swap: Boolean(getModelProfile(config.model.architecture).supportsBlockSwap),
      comfy: Boolean(getModelProfile(config.model.architecture).comfy),
      text_encoder_array: Boolean(getModelProfile(config.model.architecture).textEncoderArray),
      text_encoder_type: getModelProfile(config.model.architecture).textEncoderType || '',
      required_paths: getModelProfile(config.model.architecture).fields
        .filter((field) => field.required)
        .map((field) => ({ key: field.key, kind: field.kind, label: pick(lang, field.zh, field.en) })),
      dtype: config.model.dtype,
      transformer_dtype: config.model.transformerDtype,
      timestep_sample_method: config.model.timestepSampleMethod,
    },
    adapter: {
      type: config.adapter.type,
      rank: config.adapter.rank,
      decompose_factor: config.adapter.decomposeFactor,
      dropout: config.adapter.dropout,
      lr: config.adapter.learningRate,
      unet_lr: config.adapter.unetLearningRate,
      text_encoder_lr: config.adapter.textEncoderLearningRate,
      text_encoder_1_lr: config.adapter.textEncoderLearningRate,
      text_encoder_2_lr: config.adapter.textEncoder2LearningRate,
      init_from: config.adapter.initFrom,
    },
    training: {
      output_folder_name: config.training.outputFolderName,
      output_path: config.training.outputPath,
      epochs: config.training.epochs,
      micro_batch_size_per_gpu: config.training.microBatchSizePerGpu,
      gradient_accumulation_steps: config.training.gradientAccumulationSteps,
      warmup_steps: config.training.warmupSteps,
      blocks_to_swap: config.training.blocksToSwap,
      activation_checkpointing: config.training.activationCheckpointing,
      save_dtype: config.training.saveDtype,
      optimizer_config: JSON.stringify({ type: config.training.optimizer, lr: config.adapter.learningRate, scheduler: config.training.lrScheduler }),
      pipeline_stages: config.resources.pipelineStages,
      num_gpus: config.resources.gpuCount,
      partition_method: config.resources.partitionMethod,
      caching_batch_size: config.resources.cachingBatchSize,
      save_every_n_steps: config.training.checkpointInterval,
      steps_per_print: config.training.logInterval,
    },
    start_args: {
      resume_from_checkpoint: config.startArgs.resumeFromCheckpoint,
      regenerate_cache: config.startArgs.regenerateCache,
      trust_cache: config.startArgs.trustCache,
      cache_only: config.startArgs.cacheOnly,
      reset_dataloader: config.startArgs.resetDataloader,
      reset_optimizer_params: config.startArgs.resetOptimizerParams,
    },
  }), [config, resolution])

  const startTraining = useCallback(async () => {
    setLogs([])
    setTrainingHistory([])
    setTrainingEvents([])
    setTrainingMetrics({ currentStep: 0, totalSteps: 0, stepSeconds: null, elapsedSeconds: 0, remainingSeconds: null, outputDir: '', rawLogPath: '', estimated: true, startedAt: Date.now(), epoch: null, loss: null, learningRate: '' })
    setRuntime({ running: true, status: 'starting', label: t('正在启动', 'Starting'), message: t('正在连接训练核心…', 'Connecting to the training core…'), progress: 2, pid: null })
    if (desktop) {
      try {
        const monitorResult = await desktop.monitor.start().catch(() => ({ success: false }))
        setMonitorRunning(Boolean(monitorResult.success))
        const result = await desktop.training.start(backendConfig)
        if (!result.success) {
          setRuntime({ running: false, status: 'error', label: t('无法启动', 'Could not start'), message: result.message || t('后端未就绪。', 'Backend is not ready.'), progress: 0, pid: null })
          showToast(result.message || t('启动失败', 'Launch failed'))
          return
        }
        setRuntime((current) => ({ ...current, running: true, status: 'starting', label: t('正在启动', 'Starting'), message: t('训练进程已启动，等待训练核心确认…', 'Training process started; waiting for core confirmation…'), pid: result.pid }))
      } catch (error) {
        setRuntime({ running: false, status: 'error', label: t('无法启动', 'Could not start'), message: error.message, progress: 0, pid: null })
      }
      return
    }

    setRuntime({ running: false, status: 'error', label: t('仅支持桌面版', 'Desktop required'), message: t('浏览器预览不会模拟训练，请用 Windows EXE 连接真实核心。', 'Browser preview never simulates training; use the Windows EXE with the real core.'), progress: 0, pid: null })
  }, [backendConfig, config.model.architecture, resolution, showToast, t])

  const browseTaggerFolder = useCallback(async () => {
    if (!desktop) return showToast(t('请在桌面应用中选择目录。', 'Choose the folder in the desktop app.'))
    const selected = await desktop.dialog.folder({ title: t('选择需要自动打标的图片目录', 'Choose images to tag') })
    if (selected) setTaggerConfig((current) => ({ ...current, rootPath: selected }))
  }, [showToast, t])

  const startTagger = useCallback(async () => {
    setTaggerLogs([])
    setTaggerRuntime({ running: true, status: 'running', label: t('正在启动', 'Starting'), message: t('正在准备 WD14 模型…', 'Preparing the WD14 model…'), current: 0, total: taggerScan.total, written: 0, skipped: 0, failed: 0 })
    if (!desktop) return setTaggerRuntime((current) => ({ ...current, running: false, status: 'completed', label: t('预览完成', 'Preview complete'), message: t('桌面版会执行真实打标。', 'The desktop app performs real tagging.') }))
    const result = await desktop.tagger.start(taggerConfig).catch((error) => ({ success: false, message: error.message }))
    if (!result.success) setTaggerRuntime((current) => ({ ...current, running: false, status: 'error', label: t('无法启动', 'Could not start'), message: result.message }))
  }, [taggerConfig, taggerScan.total, t])

  const stopTagger = useCallback(async () => { await desktop?.tagger?.stop().catch(() => {}) }, [])

  const openOutput = useCallback(async () => {
    if (!trainingMetrics.outputDir) return
    const result = await desktop?.system?.openPath(trainingMetrics.outputDir)
    if (result) showToast(result)
  }, [trainingMetrics.outputDir, showToast])

  const stopTraining = useCallback(async () => {
    if (desktop) await desktop.training.stop().catch(() => {})
    setRuntime((current) => ({ ...current, running: false, status: 'stopped', label: t('已停止', 'Stopped'), message: t('训练已停止。', 'Training stopped.') }))
  }, [t])

  const toggleMonitor = useCallback(async () => {
    if (!desktop) {
      setMonitorStats({
        cpu_model: 'Preview CPU',
        cpu_percent: 18,
        memory: { total: 32 * 1024 ** 3, used: 11 * 1024 ** 3, percent: 34 },
        gpus: [{ name: 'Preview GPU', gpu_util: 12, mem_total: 24 * 1024 ** 3, mem_used: 4.6 * 1024 ** 3, temperature: 43 }],
        disks: [{ device: 'C:', mountpoint: 'C:\\', percent: 61 }],
      })
      setMonitorRunning((current) => !current)
      return
    }
    if (monitorRunning) {
      await desktop.monitor.stop()
      setMonitorRunning(false)
    } else {
      const result = await desktop.monitor.start()
      setMonitorRunning(Boolean(result.success))
      if (!result.success) showToast(result.message)
    }
  }, [monitorRunning, showToast])

  const refreshDiagnostics = useCallback(async () => {
    if (!desktop) {
      setDiagnostics({ appVersion: '0.1.0', platform: navigator.platform, python: 'Desktop only', pythonVersion: 'Open in Electron to inspect', pythonReady: false, bridge: 'backend/backend_bridge.py', bridgeReady: true, monitor: 'backend/monitor.py', monitorReady: true, trainerRoots: [] })
      return
    }
    const result = await desktop.system.diagnostics()
    setDiagnostics(result)
    if (!trainerRoot && result.trainerRoot) setTrainerRootState(result.trainerRoot)
  }, [trainerRoot])

  useEffect(() => {
    if (page === 'diagnostics' && !diagnostics) refreshDiagnostics()
  }, [page, diagnostics, refreshDiagnostics])

  const setTrainerRoot = useCallback((value) => {
    setTrainerRootState(value)
    savePreference({ trainerRoot: value })
  }, [savePreference])

  const browseTrainerRoot = useCallback(async () => {
    if (!desktop) return showToast(t('请在桌面应用中选择目录。', 'Choose the folder in the desktop app.'))
    const selected = await desktop.dialog.folder({ title: t('选择原整合包根目录', 'Choose the original trainer root') })
    if (!selected) return
    setTrainerRoot(selected)
    window.setTimeout(refreshDiagnostics, 120)
  }, [refreshDiagnostics, setTrainerRoot, showToast, t])

  const openGoofish = useCallback(() => {
    if (desktop) desktop.system.openExternal(GOOFISH_URL)
    else window.open(GOOFISH_URL, '_blank', 'noopener,noreferrer')
  }, [])

  const pageProps = { lang, config, updateSection, browseFolder, browseFile, browseModelPath, resolution }
  let pageContent
  if (page === 'home') pageContent = <HomePage lang={lang} setPage={setPage} config={config} resolution={resolution} humanized />
  else if (page === 'dataset') pageContent = <DatasetPage {...pageProps} datasetStats={datasetStats} datasetScanning={datasetScanning} />
  else if (page === 'tagger') pageContent = <WD14Page lang={lang} config={taggerConfig} setConfig={setTaggerConfig} runtime={taggerRuntime} logs={taggerLogs} scan={taggerScan} diagnostics={taggerDiagnostics} browseFolder={browseTaggerFolder} startTagger={startTagger} stopTagger={stopTagger} />
  else if (page === 'training') pageContent = <TrainingPage {...pageProps} setPage={setPage} resetTraining={resetTraining} />
  else if (page === 'start') pageContent = <StartPage lang={lang} config={config} updateSection={updateSection} resolution={resolution} datasetStats={datasetStats} backendConfig={backendConfig} runtime={runtime} metrics={trainingMetrics} monitorStats={monitorStats} startTraining={startTraining} stopTraining={stopTraining} openOutput={openOutput} logs={logs} clearLogs={() => setLogs([])} history={trainingHistory} events={trainingEvents} humanized logLevel={consoleLogLevel} setLogLevel={setConsoleLogLevel} setPage={setPage} />
  else if (page === 'resources') pageContent = <ResourcePage lang={lang} config={config} updateSection={updateSection} monitorStats={monitorStats} monitorRunning={monitorRunning} toggleMonitor={toggleMonitor} />
  else if (page === 'license') pageContent = <LicensePage lang={lang} status={license} refresh={refreshLicense} showToast={showToast} />
  else pageContent = <DiagnosticsPage lang={lang} diagnostics={diagnostics} refreshDiagnostics={refreshDiagnostics} trainerRoot={trainerRoot} setTrainerRoot={setTrainerRoot} browseTrainerRoot={browseTrainerRoot} />

  return (
    <div className={`app theme-${theme} ${glass ? 'has-glass' : 'no-glass'} ${sidebarCollapsed ? 'is-sidebar-collapsed' : ''}`}>
      <div className="ambient" aria-hidden="true"><span className="ambient__wash ambient__wash--one" /><span className="ambient__wash ambient__wash--two" /><span className="ambient__wash ambient__wash--three" /><span className="ambient__caustic" /></div>
      <svg className="svg-filters" aria-hidden="true">
        <defs>
          <filter id="liquid-edge" x="-20%" y="-20%" width="140%" height="140%">
            <feTurbulence type="fractalNoise" baseFrequency="0.008 0.018" numOctaves="2" seed="4" result="noise" />
            <feGaussianBlur in="noise" stdDeviation="0.6" result="softNoise" />
            <feDisplacementMap in="SourceGraphic" in2="softNoise" scale="11" xChannelSelector="R" yChannelSelector="B" />
          </filter>
        </defs>
      </svg>

      <header className="titlebar">
        <div className="titlebar__brand"><BrandMark /><strong>{t('特异人士的 LoRA 训练器', 'Exceptional LoRA Trainer')}</strong><span className="version-badge">0.5.2</span></div>
        <div className="titlebar__drag" />
        <div className="titlebar__status"><span className="status-dot" />{desktop ? t('桌面后端已连接', 'Desktop bridge connected') : t('前端预览', 'Frontend preview')}</div>
        <div className="window-controls">
          <button type="button" onClick={() => desktop?.window.minimize()}><Minimize2 size={16} /></button>
          <button type="button" onClick={() => desktop?.window.toggleMaximize()}><Square size={14} /></button>
          <button type="button" className="window-controls__close" onClick={() => desktop?.window.close()}><X size={17} /></button>
        </div>
      </header>

      <aside className="sidebar glass-rail">
        <button type="button" className="sidebar-collapse" onClick={() => setSidebarCollapsed((current) => !current)}><PanelLeftClose size={17} /></button>
        <nav>
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <button key={item.id} type="button" className={`nav-item ${page === item.id ? 'is-active' : ''}`} onClick={() => setPage(item.id)} title={pick(lang, item.zh, item.en)}>
                <IconWell><Icon size={20} /></IconWell><span>{pick(lang, item.zh, item.en)}</span>{page === item.id && <i />}
              </button>
            )
          })}
        </nav>
        <div className="sidebar__bottom">
          <button type="button" className="utility-row" onClick={() => { setGlass((current) => { const next = !current; savePreference({ glass: next }); return next }) }}><IconWell size="small"><Sparkles size={16} /></IconWell><span>{t('液态玻璃', 'Liquid glass')}</span><span className={`switch ${glass ? 'is-on' : ''}`}><span /></span></button>
          <button type="button" className="utility-row" onClick={openGoofish}><IconWell size="small"><ExternalLink size={16} /></IconWell><span>{t('闲鱼主页', 'Goofish')}</span><ChevronRight size={15} /></button>
          <button type="button" className="utility-row" onClick={() => { const next = lang === 'zh' ? 'en' : 'zh'; setLang(next); savePreference({ language: next }) }}><IconWell size="small"><Languages size={16} /></IconWell><span>{lang === 'zh' ? '中 / EN' : 'EN / 中'}</span><ChevronRight size={15} /></button>
          <button type="button" className="utility-row" onClick={() => { const next = theme === 'light' ? 'dark' : 'light'; setTheme(next); savePreference({ theme: next }) }}><IconWell size="small">{theme === 'light' ? <Sun size={16} /> : <Moon size={16} />}</IconWell><span>{theme === 'light' ? t('浅色模式', 'Light mode') : t('深色模式', 'Dark mode')}</span><span className="theme-dot" /></button>
        </div>
      </aside>

      <main className="content"><div className="content__scroll" key={page}><div className="page-transition">{pageContent}</div></div></main>
      {license && !license.active && !licenseGateDismissed && <LicenseGate lang={lang} status={license} onLater={()=>setLicenseGateDismissed(true)} onOpen={()=>{setLicenseGateDismissed(true);setPage('license')}} />}
      {toast && <div className="toast"><Info size={17} /><span>{toast}</span></div>}
    </div>
  )
}

export default App
