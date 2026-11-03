import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'

// Only this public key is shipped in the customer application. The matching
// private key lives with the seller and is used solely by the issuer utility.
export const LICENSE_PUBLIC_KEY = `-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEADdPQ0aseiqUVh8oEyPYBekdM58vqBzIhakylZbcv8No=
-----END PUBLIC KEY-----`

const canonical = (value) => JSON.stringify(value, Object.keys(value).sort())
const b64url = (value) => Buffer.from(value).toString('base64url')
const unb64url = (value) => Buffer.from(value, 'base64url')

function machineGuid() {
  if (process.platform !== 'win32') return ''
  const result = spawnSync('reg', ['query', 'HKLM\\SOFTWARE\\Microsoft\\Cryptography', '/v', 'MachineGuid'], { encoding: 'utf8', windowsHide: true })
  return (result.stdout || '').match(/MachineGuid\s+REG_\w+\s+(.+)/i)?.[1]?.trim() || ''
}

export function machineHash() {
  const parts = [machineGuid(), os.hostname(), os.platform(), os.arch()].filter(Boolean).join('|')
  return crypto.createHash('sha256').update(`ExceptionalLoRATrainer:v1|${parts}`, 'utf8').digest('hex')
}

function keyPaths(userData) {
  const dir = path.join(userData, 'license')
  return { dir, privateKey: path.join(dir, 'device-private.pem'), publicKey: path.join(dir, 'device-public.pem'), license: path.join(dir, 'license.lic') }
}

function deviceKeys(userData) {
  const paths = keyPaths(userData)
  fs.mkdirSync(paths.dir, { recursive: true })
  if (!fs.existsSync(paths.privateKey) || !fs.existsSync(paths.publicKey)) {
    const keys = crypto.generateKeyPairSync('ed25519')
    fs.writeFileSync(paths.privateKey, keys.privateKey.export({ type: 'pkcs8', format: 'pem' }), { mode: 0o600 })
    fs.writeFileSync(paths.publicKey, keys.publicKey.export({ type: 'spki', format: 'pem' }))
  }
  return { privateKey: fs.readFileSync(paths.privateKey, 'utf8'), publicKey: fs.readFileSync(paths.publicKey, 'utf8'), paths }
}

export function requestCode(userData) {
  const { publicKey } = deviceKeys(userData)
  return b64url(JSON.stringify({ v: 1, product: 'Exceptional-LoRA-Trainer', machineHash: machineHash(), devicePublicKey: publicKey }))
}

export function verifyLicenseObject(license, userData) {
  if (!license?.payload || !license?.signature) return { valid: false, reason: '许可证格式无效。' }
  const signatureOk = crypto.verify(null, Buffer.from(canonical(license.payload)), LICENSE_PUBLIC_KEY, unb64url(license.signature))
  if (!signatureOk) return { valid: false, reason: '许可证签名无效。' }
  const { publicKey, privateKey } = deviceKeys(userData)
  if (license.payload.machineHash !== machineHash()) return { valid: false, reason: '许可证绑定到另一台设备。' }
  if (license.payload.devicePublicKey !== publicKey) return { valid: false, reason: '本机设备密钥与许可证不匹配。' }
  if (license.payload.expiresAt && Date.parse(license.payload.expiresAt) < Date.now()) return { valid: false, reason: '许可证已过期。' }
  const nonce = crypto.randomBytes(32)
  const response = crypto.sign(null, nonce, privateKey)
  if (!crypto.verify(null, nonce, publicKey, response)) return { valid: false, reason: '本机密钥挑战验证失败。' }
  return { valid: true, licenseId: license.payload.licenseId, customer: license.payload.customer || '', edition: license.payload.edition || 'commercial', expiresAt: license.payload.expiresAt || null }
}

export function status(userData) {
  const { paths } = deviceKeys(userData)
  if (!fs.existsSync(paths.license)) return { active: false, requestCode: requestCode(userData), reason: '尚未导入许可证。' }
  try {
    const result = verifyLicenseObject(JSON.parse(fs.readFileSync(paths.license, 'utf8')), userData)
    return { active: result.valid, requestCode: result.valid ? '' : requestCode(userData), ...result }
  } catch { return { active: false, requestCode: requestCode(userData), reason: '许可证文件无法读取。' } }
}

export function activate(licenseText, userData) {
  let license
  try { license = JSON.parse(licenseText) } catch { try { license = JSON.parse(Buffer.from(String(licenseText).trim(), 'base64url').toString('utf8')) } catch { return { active: false, reason: '许可证码无效。' } } }
  const result = verifyLicenseObject(license, userData)
  if (!result.valid) return { active: false, ...result }
  const { paths } = deviceKeys(userData)
  fs.writeFileSync(paths.license, JSON.stringify(license, null, 2), { mode: 0o600 })
  return { active: true, ...result }
}
