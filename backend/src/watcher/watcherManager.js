const chokidar = require('chokidar');
const worker = require('../lib/workerClient');
const db = require('../db');

const watchers = new Map();

// Only watch file types that the worker can actually index
const SUPPORTED_EXTENSIONS = /\.(txt|md|py|js|ts|json|csv|log|pdf|docx|png|jpg|jpeg|tiff)$/i;

function startWatcher(dirPath) {
  if (watchers.has(dirPath)) return;
  const watcher = chokidar.watch(dirPath, {
    persistent: true,
    ignoreInitial: true,
    depth: 5,
    usePolling: true,
    interval: 2000,
    binaryInterval: 5000,
    ignored: [
      '**/node_modules/**',
      '**/.git/**',
      '**/.venv/**',
      '**/venv/**',
      '**/__pycache__/**',
      '**/dist/**',
      '**/build/**',
      '**/.DS_Store',
      '**/Thumbs.db',
      '**/\.Trash/**',
      '**/Library/**',
      '**/\.cache/**',
      '**/\.npm/**',
      '**/\.yarn/**',
      '**/\.next/**',
      '**/coverage/**',
      '**/tmp/**',
      '**/temp/**',
    ],
    awaitWriteFinish: {
      stabilityThreshold: 1500,
      pollInterval: 200,
    },
  });

  // Throttle error logging - only log once every 10 seconds
  let lastErrorTime = 0;
  let errorCount = 0;
  watcher.on('error', error => {
    errorCount++;
    const now = Date.now();
    if (now - lastErrorTime > 10000) {
      console.warn(`⚠️ 👀 Watcher error (${errorCount} total): ${error.message}`);
      lastErrorTime = now;
      errorCount = 0;
    }
  });

  watcher
    .on('add', async path => {
      if (!SUPPORTED_EXTENSIONS.test(path)) return;
      console.log('📄 ➕ New file detected:', path);
      try {
        const res = await worker.indexFile(path);
        console.log('✅ 🔍 File indexed successfully:', path.split(/[\\/]/).pop());
        // optional: update DB using res info
        db.upsertFile({ path, dir_id: null, checksum: res.data.checksum || null, last_indexed: new Date().toISOString(), status: 'indexed' });
      } catch (e) { 
        console.error('❌ 🔍 File indexing failed:', e.message); 
      }
    })
    .on('change', async path => {
      if (!SUPPORTED_EXTENSIONS.test(path)) return;
      console.log('📄 ✏️  File changed:', path);
      try {
        const res = await worker.reindexFile(path);
        console.log('✅ 🔄 File reindexed successfully:', path.split(/[\\/]/).pop());
        db.upsertFile({ path, dir_id: null, checksum: res.data.checksum || null, last_indexed: new Date().toISOString(), status: 'indexed' });
      } catch (e) { 
        console.error('❌ 🔄 File reindexing failed:', e.message); 
      }
    })
    .on('unlink', async path => {
      if (!SUPPORTED_EXTENSIONS.test(path)) return;
      console.log('📄 🗑️  File deleted:', path);
      try {
        await worker.removeFile(path);
        console.log('✅ 🗑️  File removed from index:', path.split(/[\\/]/).pop());
        db.removeFile(path);
      } catch (e) { 
        console.error('❌ 🗑️  File removal failed:', e.message); 
      }
    });

  watchers.set(dirPath, watcher);
  console.log('🎯 👀 File watcher activated for:', dirPath);
}

function stopWatcher(dirPath) {
  const w = watchers.get(dirPath);
  if (w) {
    w.close();
    watchers.delete(dirPath);
  }
}

function status() {
  return Array.from(watchers.keys());
}

module.exports = { startWatcher, stopWatcher, status };
