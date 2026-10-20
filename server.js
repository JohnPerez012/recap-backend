/**
 * RECAP Backend Server
 * Enhanced RAG + 4-Tier AI Fallback System
 * 
 * AI Priority: Mistral → Groq → Gemini → OpenRouter
 * Pinecone: Semantic search with multilingual-e5-large embeddings
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const admin = require('firebase-admin');
const { Pinecone } = require('@pinecone-database/pinecone');
const fs = require('fs');
let cloudinary = null;
try {
  cloudinary = require('cloudinary').v2;
} catch (err) {
  console.warn('⚠️ Cloudinary module not found. Image deletion endpoint will be disabled until installed.');
}

// Polyfill fetch for Node.js < 18
if (typeof fetch === 'undefined') {
  global.fetch = require('node-fetch');
}

// Configure undici with 30s connection timeout to eliminate ConnectTimeoutError to Pinecone
try {
  const { Agent, setGlobalDispatcher } = require('undici');
  setGlobalDispatcher(new Agent({ connect: { timeout: 30000 } }));
  console.log('✓ Undici global dispatcher configured with 30s connect timeout');
} catch (err) {
  console.warn('⚠️ Could not configure undici dispatcher:', err.message);
}

const app = express();
const PORT = process.env.PORT || 3001;

// ============================================
// MIDDLEWARE
// ============================================
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));
app.options('*', cors());
app.use(express.json());
app.use(express.static('public'));

// ============================================
// FIREBASE INITIALIZATION
// ============================================
let serviceAccount;

if (process.env.FIREBASE_SERVICE_ACCOUNT_JSON) {
  try {
    // Parse JSON directly from Render Environment Variable
    serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON);
    console.log('✓ Using Firebase credentials from environment variable (Render)');
  } catch (error) {
    console.error('❌ Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON environment variable:', error.message);
    process.exit(1);
  }
} else {
  // Fallback to local file for development
  const serviceAccountPath = process.env.FIREBASE_SERVICE_ACCOUNT_PATH || './serviceAccountKey.json';
  if (fs.existsSync(serviceAccountPath)) {
    console.log(`Using Firebase credentials from file: ${serviceAccountPath}`);
    serviceAccount = require(serviceAccountPath);
  } else {
    console.error('❌ Firebase service account not found in environment variables or file path:', serviceAccountPath);
    process.exit(1);
  }
}

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});
console.log('✓ Firebase Admin initialized successfully');

const db = admin.firestore();

// ============================================
// PINECONE INITIALIZATION
// ============================================
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});

const indexName = 'recaps-projects-search';
const index = pinecone.index(indexName, process.env.PINECONE_INDEX_HOST);

console.log(`✓ Pinecone index: ${indexName}`);
console.log('✓ Using Inference API (automatic embeddings)');

// ============================================
// RESEND EMAIL CONFIGURATION
// ============================================
const RESEND_API_KEY = process.env.RESEND_API_KEY ? process.env.RESEND_API_KEY.trim() : '';
const RESEND_FROM_EMAIL = process.env.RESEND_FROM_EMAIL ? process.env.RESEND_FROM_EMAIL.trim() : 'onboarding@resend.dev';
const RESEND_FROM_NAME = process.env.RESEND_FROM_NAME ? process.env.RESEND_FROM_NAME.trim() : 'RECAP Support';
const FEEDBACK_RECIPIENT_EMAIL = process.env.FEEDBACK_RECIPIENT_EMAIL ? process.env.FEEDBACK_RECIPIENT_EMAIL.trim() : '';

if (RESEND_API_KEY) {
  console.log('✓ Resend API configured');
  console.log(`  From: ${RESEND_FROM_NAME} <${RESEND_FROM_EMAIL}>`);
  if (FEEDBACK_RECIPIENT_EMAIL) {
    console.log(`  Admin Recipient: ${FEEDBACK_RECIPIENT_EMAIL}`);
  } else {
    console.log('  Admin Recipient: (Disabled - saving 1 email quota per feedback)');
  }
} else {
  console.warn('⚠️  Resend API key not configured. Feedback emails will not be sent.');
}

// ============================================
// CLOUDINARY CONFIGURATION
// ============================================
try {
  if (cloudinary && process.env.CLOUDINARY_CLOUD_NAME && process.env.CLOUDINARY_API_KEY && process.env.CLOUDINARY_API_SECRET) {
    cloudinary.config({
      cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
      api_key: process.env.CLOUDINARY_API_KEY,
      api_secret: process.env.CLOUDINARY_API_SECRET
    });
    console.log('✓ Cloudinary configured');
  } else if (!cloudinary) {
    console.warn('⚠️  Cloudinary package not available. Image deletion disabled.');
  } else {
    console.warn('⚠️  Cloudinary credentials not configured. Image deletion will not work.');
  }
} catch (error) {
  console.error('❌ Failed to configure Cloudinary:', error.message);
}

// ============================================
// AI PROVIDER CONFIGURATIONS
// ============================================
const AI_PROVIDERS = { 
  mistral: { // first
    name: 'Mistral AI',
    endpoint: 'https://api.mistral.ai/v1/chat/completions',
    streamEndpoint: 'https://api.mistral.ai/v1/chat/completions',
    model: 'open-mistral-nemo',
    apiKey: process.env.MISTRAL_API_KEY,
    headers: (apiKey) => ({
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    }),
    formatRequest: (messages, maxTokens, stream = false) => ({
      model: 'open-mistral-nemo',
      messages,
      max_tokens: maxTokens,
      temperature: 0.7,
      stream: !!stream
    }),
    extractResponse: (data) => data.choices[0].message.content,
    extractStreamDelta: (json) => json.choices?.[0]?.delta?.content || ''
  },
  
  groq: { // Second
    name: 'Groq AI',
    endpoint: 'https://api.groq.com/openai/v1/chat/completions',
    streamEndpoint: 'https://api.groq.com/openai/v1/chat/completions',
    model: 'llama-3.3-70b-versatile',
    apiKey: process.env.GROQ_API_KEY,
    headers: (apiKey) => ({
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    }),
    formatRequest: (messages, maxTokens, stream = false) => ({
      model: 'llama-3.3-70b-versatile',
      messages,
      max_tokens: maxTokens,
      temperature: 0.7,
      stream: !!stream
    }),
    extractResponse: (data) => data.choices[0].message.content,
    extractStreamDelta: (json) => json.choices?.[0]?.delta?.content || ''
  },
  
  gemini: { // Third
    name: 'Google Gemini',
    endpoint: (apiKey) => `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=${apiKey}`,
    streamEndpoint: (apiKey) => `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse&key=${apiKey}`,
    model: 'gemini-3.5-flash',
    apiKey: process.env.GEMINI_API_KEY,
    headers: () => ({
      'Content-Type': 'application/json'
    }),
    formatRequest: (messages, maxTokens) => {
      // Extract system message if present
      const systemMessage = messages.find(msg => msg.role === 'system');
      const nonSystemMessages = messages.filter(msg => msg.role !== 'system');

      const contents = nonSystemMessages.map(msg => ({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content }]
      }));
      
      const payload = {
        contents,
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: maxTokens
        }
      };

      if (systemMessage) {
        payload.systemInstruction = {
          parts: [{ text: systemMessage.content }]
        };
      }

      return payload;
    },
    extractResponse: (data) => data.candidates[0].content.parts[0].text,
    extractStreamDelta: (json) => json.candidates?.[0]?.content?.parts?.[0]?.text || ''
  },
  
  openrouter: { // fourth
    name: 'OpenRouter',
    endpoint: 'https://openrouter.ai/api/v1/chat/completions',
    streamEndpoint: 'https://openrouter.ai/api/v1/chat/completions',
    model: 'inclusionai/ling-3.0-flash-sante:free',
    apiKey: process.env.OPENROUTER_API_KEY,
    headers: (apiKey) => ({
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://recaps-project-hub.onrender.com',
      'X-Title': 'RECAP Project Hub'
    }),
    formatRequest: (messages, maxTokens, stream = false) => ({
      model: 'inclusionai/ling-3.0-flash-sante:free',
      messages,
      max_tokens: maxTokens,
      temperature: 0.7,
      stream: !!stream
    }),
    extractResponse: (data) => data.choices?.[0]?.message?.content || data.choices?.[0]?.message?.reasoning || '',
    extractStreamDelta: (json) => json.choices?.[0]?.delta?.content || json.choices?.[0]?.delta?.reasoning || ''
  }
};

// ============================================
// AI HELPER FUNCTIONS
// ============================================

/**
 * Async generator to read Server-Sent Events lines from a ReadableStream or Node stream
 */
async function* readSSELines(responseBody) {
  let buffer = '';
  const decoder = new TextDecoder('utf-8');

  if (responseBody.getReader) {
    const reader = responseBody.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          yield line;
        }
      }
    } finally {
      if (typeof reader.releaseLock === 'function') {
        reader.releaseLock();
      }
    }
  } else if (responseBody[Symbol.asyncIterator]) {
    for await (const chunk of responseBody) {
      buffer += typeof chunk === 'string' ? chunk : decoder.decode(chunk, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        yield line;
      }
    }
  }

  if (buffer.trim()) {
    yield buffer;
  }
}

/**
 * Call AI provider for a non-streaming response
 */
async function callAIProvider(providerKey, messages, maxTokens = 1000) {
  const provider = AI_PROVIDERS[providerKey];
  
  if (!provider.apiKey) {
    throw new Error(`${provider.name} API key not configured`);
  }
  
  try {
    const endpoint = typeof provider.endpoint === 'function' 
      ? provider.endpoint(provider.apiKey)
      : provider.endpoint;
    
    const headers = provider.headers(provider.apiKey);
    const body = provider.formatRequest(messages, maxTokens, false);
    
    console.log(`🤖 Calling ${provider.name}...`);
    
    const response = await fetch(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(body)
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`${provider.name} error (${response.status}): ${errorText}`);
    }
    
    const data = await response.json();
    const content = provider.extractResponse(data);
    
    console.log(`✓ ${provider.name} responded successfully`);
    return content;
    
  } catch (error) {
    console.error(`❌ ${provider.name} failed:`, error.message);
    throw error;
  }
}

/**
 * Stream tokens from a single AI provider via SSE
 */
async function streamFromAIProvider(providerKey, messages, maxTokens = 1000, onToken) {
  const provider = AI_PROVIDERS[providerKey];
  
  if (!provider.apiKey) {
    throw new Error(`${provider.name} API key not configured`);
  }
  
  const endpoint = typeof provider.streamEndpoint === 'function' 
    ? provider.streamEndpoint(provider.apiKey)
    : (provider.streamEndpoint || provider.endpoint);
  
  const headers = provider.headers(provider.apiKey);
  const body = provider.formatRequest(messages, maxTokens, true);
  
  console.log(`🤖 Streaming from ${provider.name}...`);
  
  const response = await fetch(endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify(body)
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`${provider.name} streaming error (${response.status}): ${errorText}`);
  }
  
  let fullText = '';
  let tokenCount = 0;

  for await (const line of readSSELines(response.body)) {
    const trimmed = line.trim();
    if (!trimmed || !trimmed.startsWith('data:')) continue;
    
    const dataStr = trimmed.slice(5).trim();
    if (dataStr === '[DONE]') break;
    
    try {
      const json = JSON.parse(dataStr);
      const delta = provider.extractStreamDelta ? provider.extractStreamDelta(json) : '';
      if (delta) {
        tokenCount++;
        fullText += delta;
        if (typeof onToken === 'function') {
          onToken(delta);
        }
      }
    } catch (e) {
      // Skip non-JSON or partial keepalive lines
    }
  }

  console.log(`✓ ${provider.name} stream completed (${tokenCount} chunks, ${fullText.length} chars)`);
  return fullText;
}

/**
 * Resolves the provider cascade order based on user preference
 */
function getProviderSequence(preferredKey) {
  const defaultOrder = ['mistral', 'groq', 'gemini', 'openrouter'];
  if (!preferredKey) return defaultOrder;
  const normalized = String(preferredKey).toLowerCase().replace(/\s+/g, '');
  let match = null;
  if (normalized.includes('mistral')) match = 'mistral';
  else if (normalized.includes('groq')) match = 'groq';
  else if (normalized.includes('gemini')) match = 'gemini';
  else if (normalized.includes('openrouter')) match = 'openrouter';
  if (!match) return defaultOrder;
  return [match, ...defaultOrder.filter(k => k !== match)];
}

/**
 * 4-Tier cascading fallback system with real-time SSE streaming
 */
async function streamAIWithFallback(messages, maxTokens = 1000, res, metadata = {}) {
  const providers = getProviderSequence(metadata.preferredProvider);
  const errors = [];
  
  for (let i = 0; i < providers.length; i++) {
    const providerKey = providers[i];
    const provider = AI_PROVIDERS[providerKey];
    
    if (!provider.apiKey) {
      errors.push({ provider: provider.name, error: 'API key not configured' });
      continue;
    }
    
    let hasSentStart = false;
    let fullText = '';
    
    try {
      await streamFromAIProvider(providerKey, messages, maxTokens, (token) => {
        if (!hasSentStart) {
          hasSentStart = true;
          // Send start event to client
          res.write(`data: ${JSON.stringify({
            type: 'start',
            provider: provider.name,
            providerKey,
            attemptNumber: i + 1,
            ...metadata
          })}\n\n`);
        }
        
        fullText += token;
        res.write(`data: ${JSON.stringify({
          type: 'token',
          content: token
        })}\n\n`);
        
        if (typeof res.flush === 'function') {
          res.flush();
        }
      });
      
      // Successfully finished streaming full response
      res.write(`data: ${JSON.stringify({
        type: 'done',
        fullResponse: fullText,
        provider: provider.name,
        providerKey,
        attemptNumber: i + 1,
        ...metadata
      })}\n\n`);
      res.end();
      return;
      
    } catch (error) {
      console.error(`❌ ${provider.name} streaming error:`, error.message);
      errors.push({
        provider: provider.name,
        error: error.message
      });
      
      if (hasSentStart) {
        // Interrupted mid-stream after emitting start/tokens
        res.write(`data: ${JSON.stringify({
          type: 'error',
          error: `${provider.name} stream interrupted: ${error.message}`
        })}\n\n`);
        res.end();
        return;
      }
      
      // If no tokens were sent yet, try next provider
      if (i < providers.length - 1) {
        console.log(`⚠️  ${provider.name} failed before sending stream, cascading to next provider...`);
        continue;
      }
    }
  }
  
  // All providers failed
  res.write(`data: ${JSON.stringify({
    type: 'error',
    error: 'All AI providers are currently unavailable. Please try again later.',
    details: errors
  })}\n\n`);
  res.end();
}

/**
 * 4-Tier cascading fallback system (Non-streaming)
 */
async function callAIWithFallback(messages, maxTokens = 1000, preferredKey = null) {
  const providers = getProviderSequence(preferredKey);
  const errors = [];
  
  for (let i = 0; i < providers.length; i++) {
    const providerKey = providers[i];
    const provider = AI_PROVIDERS[providerKey];
    
    try {
      const response = await callAIProvider(providerKey, messages, maxTokens);
      return {
        success: true,
        response,
        provider: provider.name,
        providerKey,
        attemptNumber: i + 1
      };
    } catch (error) {
      errors.push({
        provider: provider.name,
        error: error.message
      });
      
      // If not the last provider, continue to next
      if (i < providers.length - 1) {
        console.log(`⚠️  ${provider.name} failed, trying next provider...`);
        continue;
      }
    }
  }
  
  // All providers failed
  return {
    success: false,
    errors,
    message: 'All AI providers are currently unavailable. Please try again later.'
  };
}

// ============================================
// PINECONE SYNC HELPERS
// ============================================

/**
 * Generate embedding for text using Pinecone Inference API
 */
async function generateEmbedding(text) {
  try {
    const embedResponse = await fetch('https://api.pinecone.io/embed', {
      method: 'POST',
      headers: {
        'Api-Key': process.env.PINECONE_API_KEY,
        'Content-Type': 'application/json',
        'X-Pinecone-Api-Version': '2024-10'
      },
      body: JSON.stringify({
        model: 'multilingual-e5-large',
        parameters: {
          input_type: 'passage',
          truncate: 'END'
        },
        inputs: [{ text }]
      })
    });
    
    if (!embedResponse.ok) {
      const errorText = await embedResponse.text();
      throw new Error(`Embedding generation failed: ${errorText}`);
    }
    
    const embedData = await embedResponse.json();
    return embedData.data[0].values;
  } catch (error) {
    console.error('❌ Embedding generation error:', error);
    throw error;
  }
}

/**
 * Format project text with explicit semantic labels for Pinecone embeddings
 * Produces structured format:
 * "Title: ... Abstract: ... Adviser: ... Authors: ... Program: ... Year: ... Keywords: ... Key Findings: ... Topics: ..."
 */
function formatSearchableText(projectData) {
  const parts = [];

  // Title
  if (projectData.title && typeof projectData.title === 'string' && projectData.title.trim()) {
    parts.push(`Title: ${projectData.title.trim()}`);
  }

  // Abstract
  if (projectData.abstract && typeof projectData.abstract === 'string' && projectData.abstract.trim()) {
    parts.push(`Abstract: ${projectData.abstract.trim()}`);
  }

  // Adviser
  if (projectData.adviser && typeof projectData.adviser === 'string' && projectData.adviser.trim()) {
    parts.push(`Adviser: ${projectData.adviser.trim()}`);
  }

  // Authors
  const authorsStr = Array.isArray(projectData.authors)
    ? projectData.authors.map(a => (typeof a === 'string' ? a.trim() : a)).filter(Boolean).join(', ')
    : (typeof projectData.authors === 'string' ? projectData.authors.trim() : '');
  if (authorsStr) {
    parts.push(`Authors: ${authorsStr}`);
  }

  // Program
  if (projectData.program && typeof projectData.program === 'string' && projectData.program.trim()) {
    parts.push(`Program: ${projectData.program.trim()}`);
  }

  // Year
  if (projectData.year) {
    parts.push(`Year: ${projectData.year}`);
  }

  // Keywords
  const keywordsStr = Array.isArray(projectData.keywords)
    ? projectData.keywords.map(k => (typeof k === 'string' ? k.trim() : k)).filter(Boolean).join(', ')
    : (typeof projectData.keywords === 'string' ? projectData.keywords.trim() : '');
  if (keywordsStr) {
    parts.push(`Keywords: ${keywordsStr}`);
  }

  // Key Findings
  if (projectData.keyFindings && typeof projectData.keyFindings === 'string' && projectData.keyFindings.trim()) {
    parts.push(`Key Findings: ${projectData.keyFindings.trim()}`);
  }

  // Topics
  const topicsStr = Array.isArray(projectData.topics)
    ? projectData.topics.map(t => (typeof t === 'string' ? t.trim() : t)).filter(Boolean).join(', ')
    : (typeof projectData.topics === 'string' ? projectData.topics.trim() : '');
  if (topicsStr) {
    parts.push(`Topics: ${topicsStr}`);
  }

  // Ensure each segment ends with a period for clear semantic boundaries
  return parts
    .map(part => (/[.!?]$/.test(part) ? part : `${part}.`))
    .join(' ');
}

/**
 * Upsert a project to Pinecone
 */
async function upsertProjectToPinecone(projectId, projectData) {
  try {
    console.log(`📤 Upserting project to Pinecone: ${projectId}`);
    
    // Build searchable text with explicit labels for superior semantic search & context
    const searchableText = formatSearchableText(projectData);
    
    if (!searchableText) {
      console.warn('⚠️  No searchable content for project:', projectId);
      return;
    }
    
    // Generate embedding
    const embedding = await generateEmbedding(searchableText);
    
    // Normalize keywords for metadata
    const keywordsList = Array.isArray(projectData.keywords)
      ? projectData.keywords.map(k => String(k).trim()).filter(Boolean)
      : (typeof projectData.keywords === 'string' && projectData.keywords.trim()
          ? projectData.keywords.split(',').map(k => k.trim()).filter(Boolean)
          : []);

    // Prepare metadata
    const metadata = {
      title: projectData.title || 'Untitled',
      authors: Array.isArray(projectData.authors) 
        ? projectData.authors.join(', ') 
        : (projectData.authors || ''),
      program: projectData.program || '',
      year: projectData.year ? parseInt(projectData.year) : 0,
      adviser: projectData.adviser || '',
      abstract: projectData.abstract || '',
      keywords: keywordsList,
      text: searchableText,
      status: projectData.status || 'Completed'
    };
    
    // Upsert to Pinecone with retry for network resilience
    const upsertUrl = `https://${process.env.PINECONE_INDEX_HOST}/vectors/upsert`;
    let lastError = null;
    const maxAttempts = 2;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      let timeoutId = null;
      try {
        const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
        if (controller) {
          timeoutId = setTimeout(() => controller.abort(), 25000);
        }

        const response = await fetch(upsertUrl, {
          method: 'POST',
          headers: {
            'Api-Key': process.env.PINECONE_API_KEY,
            'Content-Type': 'application/json',
            'X-Pinecone-Api-Version': '2025-04'
          },
          body: JSON.stringify({
            namespace: '__default__',
            vectors: [{
              id: projectId,
              values: embedding,
              metadata
            }]
          }),
          ...(controller ? { signal: controller.signal } : {})
        });

        if (timeoutId) clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Pinecone upsert failed: ${errorText}`);
        }

        console.log(`✓ Project upserted to Pinecone: ${projectId}`);
        return true;
      } catch (err) {
        if (timeoutId) clearTimeout(timeoutId);
        lastError = err;
        console.warn(`⚠️ Pinecone upsert attempt ${attempt}/${maxAttempts} failed for ${projectId}:`, err.message);
        if (attempt < maxAttempts) {
          await new Promise(res => setTimeout(res, 1500));
        }
      }
    }

    throw lastError;
  } catch (error) {
    console.error(`❌ Failed to upsert project to Pinecone:`, error);
    throw error;
  }
}

/**
 * Delete a project from Pinecone
 */
async function deleteProjectFromPinecone(projectId) {
  try {
    console.log(`🗑️  Deleting project from Pinecone: ${projectId}`);
    
    const deleteUrl = `https://${process.env.PINECONE_INDEX_HOST}/vectors/delete`;
    const response = await fetch(deleteUrl, {
      method: 'POST',
      headers: {
        'Api-Key': process.env.PINECONE_API_KEY,
        'Content-Type': 'application/json',
        'X-Pinecone-Api-Version': '2025-04'
      },
      body: JSON.stringify({
        namespace: '__default__',
        ids: [projectId]
      })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Pinecone delete failed: ${errorText}`);
    }
    
    console.log(`✓ Project deleted from Pinecone: ${projectId}`);
    return true;
  } catch (error) {
    console.error(`❌ Failed to delete project from Pinecone:`, error);
    throw error;
  }
}

// ============================================
// PROJECT SYNC ENDPOINTS
// ============================================

/**
 * POST /api/projects/sync
 * Sync a project to Pinecone (create or update)
 */
app.post('/api/projects/sync', async (req, res) => {
  const { projectId } = req.body;
  let { projectData } = req.body;

  if (!projectId) {
    return res.status(400).json({ 
      success: false, 
      error: 'projectId is required' 
    });
  }

  // If projectData is not provided in body, load it from Firestore
  if (!projectData) {
    try {
      const projectDoc = await db.collection('projects').doc(projectId).get();
      if (!projectDoc.exists) {
        return res.status(404).json({
          success: false,
          error: `Project ${projectId} not found in database`
        });
      }
      projectData = projectDoc.data();
    } catch (dbErr) {
      console.error(`❌ Error fetching project ${projectId} from Firestore:`, dbErr);
      return res.status(500).json({
        success: false,
        error: `Database fetch failed: ${dbErr.message}`
      });
    }
  }

  try {
    await upsertProjectToPinecone(projectId, projectData);

    // Mark as synced in Firestore
    await db.collection('projects').doc(projectId).update({
      pineconeSynced: true,
      pineconeSyncedAt: admin.firestore.FieldValue.serverTimestamp()
    }).catch(updateErr => {
      console.warn(`Could not update Firestore pineconeSynced for ${projectId}:`, updateErr.message);
    });

    res.json({
      success: true,
      message: 'Project synced to Pinecone successfully',
      projectId,
      pineconeSynced: true
    });
  } catch (error) {
    console.error('❌ Sync error:', error);

    // Mark as unsynced in Firestore
    await db.collection('projects').doc(projectId).update({
      pineconeSynced: false,
      pineconeSyncError: error.message || 'Sync failed'
    }).catch(updateErr => {
      console.warn(`Could not mark unsynced for ${projectId}:`, updateErr.message);
    });

    res.status(500).json({
      success: false,
      error: error.message,
      projectId,
      pineconeSynced: false
    });
  }
});

/**
 * GET /api/projects/sync-status
 * Query Pinecone directly to get all indexed vector IDs and reconcile Firestore
 */
app.get('/api/projects/sync-status', async (req, res) => {
  try {
    const listUrl = `https://${process.env.PINECONE_INDEX_HOST}/vectors/list?namespace=__default__`;
    let vectorIds = [];
    let paginationToken = undefined;

    do {
      const url = paginationToken 
        ? `${listUrl}&paginationToken=${encodeURIComponent(paginationToken)}`
        : listUrl;

      const response = await fetch(url, {
        headers: {
          'Api-Key': process.env.PINECONE_API_KEY,
          'X-Pinecone-Api-Version': '2025-04'
        }
      });

      if (!response.ok) {
        throw new Error(`Pinecone list failed: ${response.statusText}`);
      }

      const data = await response.json();
      if (Array.isArray(data.vectors)) {
        vectorIds.push(...data.vectors.map(v => v.id));
      }
      paginationToken = data.pagination?.next;
    } while (paginationToken);

    console.log(`🔍 Verified ${vectorIds.length} vectors in Pinecone:`, vectorIds);

    // Reconcile matching Firestore documents
    for (const id of vectorIds) {
      try {
        const docRef = db.collection('projects').doc(id);
        const docSnap = await docRef.get();
        if (docSnap.exists && docSnap.data().pineconeSynced !== true) {
          await docRef.update({
            pineconeSynced: true,
            pineconeSyncedAt: admin.firestore.FieldValue.serverTimestamp()
          });
          console.log(`✓ Reconciled Firestore doc ${id} to pineconeSynced: true`);
        }
      } catch (err) {}
    }

    res.json({
      success: true,
      vectorIds,
      total: vectorIds.length
    });
  } catch (error) {
    console.error('❌ Sync-status error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * DELETE /api/projects/sync/:projectId
 * Delete a project from Pinecone
 */
app.delete('/api/projects/sync/:projectId', async (req, res) => {
  try {
    const { projectId } = req.params;
    
    if (!projectId) {
      return res.status(400).json({ 
        success: false, 
        error: 'projectId is required' 
      });
    }
    
    await deleteProjectFromPinecone(projectId);
    
    res.json({
      success: true,
      message: 'Project deleted from Pinecone successfully',
      projectId
    });
  } catch (error) {
    console.error('❌ Delete error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * POST /api/projects/sync-all
 * Sync all projects from Firestore to Pinecone
 */
app.post('/api/projects/sync-all', async (req, res) => {
  try {
    console.log('🔄 Starting full Pinecone sync...');
    
    // Fetch all projects from Firestore
    const projectsSnapshot = await db.collection('projects').get();
    const projects = [];
    
    projectsSnapshot.forEach(doc => {
      projects.push({
        id: doc.id,
        ...doc.data()
      });
    });
    
    console.log(`📦 Found ${projects.length} projects to sync`);
    
    // Sync each project to Pinecone
    const results = {
      success: 0,
      failed: 0,
      errors: []
    };
    
    for (const project of projects) {
      try {
        await upsertProjectToPinecone(project.id, project);
        results.success++;
        // Update Firestore doc with pineconeSynced: true
        await db.collection('projects').doc(project.id).update({
          pineconeSynced: true,
          pineconeSyncedAt: admin.firestore.FieldValue.serverTimestamp()
        }).catch(() => {});
      } catch (error) {
        results.failed++;
        results.errors.push({
          projectId: project.id,
          error: error.message
        });
        // Update Firestore doc with pineconeSynced: false
        await db.collection('projects').doc(project.id).update({
          pineconeSynced: false,
          pineconeSyncError: error.message
        }).catch(() => {});
      }
    }
    
    console.log(`✓ Sync complete: ${results.success} success, ${results.failed} failed`);
    
    res.json({
      success: true,
      message: 'Full sync completed',
      totalProjects: projects.length,
      synced: results.success,
      failed: results.failed,
      errors: results.errors
    });
  } catch (error) {
    console.error('❌ Full sync error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// ============================================
// RAG ENDPOINTS
// ============================================

/**
 * POST /api/pinecone/search
 * Semantic search using Pinecone Inference API
 * Uses the proven working implementation
 */
app.post('/api/pinecone/search', async (req, res) => {
  try {
    const { query, topK = 10, filter = {} } = req.body;
    
    if (!query || query.trim() === '') {
      return res.status(400).json({ error: 'Query is required' });
    }
    
    console.log(`\n🔍 RAG Search: "${query}"`);
    
    // Build metadata filter if provided
    const metadataFilter = {};
    if (filter.year !== undefined && filter.year !== null && filter.year !== '') {
      if (Array.isArray(filter.year)) {
        metadataFilter.year = { $in: filter.year.map(y => parseInt(y)) };
      } else {
        metadataFilter.year = { $eq: parseInt(filter.year) };
      }
    }
    if (filter.program) {
      if (Array.isArray(filter.program)) {
        metadataFilter.program = { $in: filter.program };
      } else {
        metadataFilter.program = { $eq: filter.program };
      }
    }
    if (filter.adviser) {
      if (Array.isArray(filter.adviser)) {
        metadataFilter.adviser = { $in: filter.adviser };
      } else {
        metadataFilter.adviser = { $eq: filter.adviser };
      }
    }
    if (filter.status) {
      if (Array.isArray(filter.status)) {
        metadataFilter.status = { $in: filter.status };
      } else {
        metadataFilter.status = { $eq: filter.status };
      }
    }
    
    // Step 1: Generate embedding for the query using Inference API
    const embedResponse = await fetch('https://api.pinecone.io/embed', {
      method: 'POST',
      headers: {
        'Api-Key': process.env.PINECONE_API_KEY,
        'Content-Type': 'application/json',
        'X-Pinecone-Api-Version': '2024-10'
      },
      body: JSON.stringify({
        model: 'multilingual-e5-large',
        parameters: {
          input_type: 'query',
          truncate: 'END'
        },
        inputs: [{ text: query }]
      })
    });
    
    if (!embedResponse.ok) {
      const errorText = await embedResponse.text();
      throw new Error(`Embedding failed (${embedResponse.status}): ${errorText}`);
    }
    
    const embedData = await embedResponse.json();
    const queryVector = embedData.data[0].values;
    
    console.log(`✓ Generated embedding (${queryVector.length} dimensions)`);
    
    // Step 2: Query Pinecone with the generated vector
    const queryUrl = `https://${process.env.PINECONE_INDEX_HOST}/query`;
    const response = await fetch(queryUrl, {
      method: 'POST',
      headers: {
        'Api-Key': process.env.PINECONE_API_KEY,
        'Content-Type': 'application/json',
        'X-Pinecone-Api-Version': '2025-04'  // Use 2025-04 for __default__ namespace
      },
      body: JSON.stringify({
        namespace: '__default__',
        vector: queryVector,
        topK,
        includeMetadata: true,
        filter: Object.keys(metadataFilter).length > 0 ? metadataFilter : undefined
      })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Query failed (${response.status}): ${errorText}`);
    }
    
    const results = await response.json();
    
    console.log(`✓ Found ${results.matches?.length || 0} results for: "${query}"`);
    
    // Transform results for frontend
    const matches = results.matches.map(match => ({
      id: match.id,
      score: match.score,
      ...match.metadata
    }));
    
    res.json({ matches });
    
  } catch (error) {
    console.error('❌ Pinecone search error:', error);
    res.status(500).json({ error: error.message });
  }
});

// ============================================
// AI CHAT ENDPOINTS
// ============================================

/**
 * POST /api/chat
 * Main chat endpoint with 4-tier fallback and response streaming
 */
app.post('/api/chat', async (req, res) => {
  try {
    const { message, conversationHistory = [], relevantProjects = [], stream = true, preferredProvider } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    console.log(`\n💬 Chat Request (stream=${stream}): "${message.substring(0, 50)}..."`);
    
    // Build context-aware prompt with enhanced academic assistant instructions
    let systemPrompt = `You are an intelligent helpful AI-Chatbot assistant for the RECAP Project Hub, versatile AI research assistant for the RECAP Project Hub, a digital repository for undergraduate theses, capstone, and academic research projects.

CRITICAL RULE: NEVER hallucinate or invent data, structures, logics, mechanism etc. Data are already fed-up and already provided by the power of RAG integration thru pinecone.

YOUR PERSONALITY:
- Friendly, calm, and conversational — like a knowledgeable classmate, not a robot.
- Use simple language. Avoid overly technical jargon unless the user seems technical.
- Keep responses short by default (2–5 sentences or a short list). If the user wants more detail, they'll ask.
- Never start your response with "Great question!" or robotic filler phrases.
- Be honest when you don't know something, but always try to help in some way.

FORMATTING:
- Use **bold** only for important labels or titles.
- Use short numbered lists only when steps are involved.
- Use bullet points only for listing multiple items.
- Avoid walls of text. Prefer short, readable paragraphs.
- Ensure all text output uses standard character encoding to avoid parsing conflicts in API payloads.

CORE GUIDELINES:
1. Ground your answers about specific research projects on the Database Context provided below from Pinecone.
2. The context includes full project titles, authors, full abstracts, programs, years, advisers, and keywords.
3. Users may ask diverse, open-ended questions in any format (e.g. asking for specific details, comparing abstract character/word lengths, findings, asking for recommendations, identifying advisers/authors, or general academic assistance).
4. When users ask questions about abstracts (such as content, character lengths, key findings, comparisons, or specific research questions), analyze the full abstracts provided in the context accurately. Full abstracts are included in the context below.
5. If the user asks for comparisons (like the one with the longest abstract, highest relevance, specific topics, or authors), analyze all the provided project records in the context and provide a precise, accurate answer.
6. Do not claim abstracts are unavailable if they are present in the context below.
7. Be polite, direct, articulate, and format your output cleanly using markdown (bullet points, bold text, numbered lists).
8. It is okay to say "I don't have that project in the current results" — that is honest and helpful.

RESPONSE STRATEGY:
9. When answering questions:
   a) If answer is in provided context → cite specific projects by title
   b) If comparing projects → show side-by-side details
   c) If asked about trends → synthesize across all provided projects
   d) If asked about specifics (methods, findings) → quote relevant abstract sections

10. Citation Format & Project Links:
    - Always mention project title when referencing data
    - Format project titles as: **[Project Title]** or [Project Title](#project)
    - FORBIDDEN: NEVER invent or hallucinate fake external URLs or web domains (e.g. do NOT write "https://recap.projecthub.org/...", "http://...", or any external URLs). All capstone projects exist within the local RE-CAPS repository.
    - Example: "According to **[Project Title]** by [Authors], ..."
    - Show relevance percentage for transparency: "(92% match)"

11. Handling Edge Cases:
    - If question needs data not in context → say "The provided projects don't contain that information. Try a more specific search query."
    - If relevance scores are low (<50%) → mention "These results have lower relevance. Consider refining your search."
    - If projects contradict each other → present both perspectives clearly

ABSTRACT ANALYSIS PROTOCOL:
12. When analyzing abstracts, identify these components if present:
    - Research Problem/Gap
    - Objectives
    - Key Findings/Results
    - Conclusions/Implications

13. For comparative analysis:
    - Create structured tables when comparing 2+ projects
    - Highlight similarities and differences

14. For length/character questions:
    - Provide exact counts
    - Explain what the length suggests (comprehensive vs. brief)`;
    // Add project count if available
    if (relevantProjects.length > 0) {
      // Check if there's a project_count metadata
      const countProject = relevantProjects.find(p => p.id && p.id.includes('document_count'));
      
      // Filter out metadata-only documents
      const actualProjects = relevantProjects.filter(p => !p.id || !p.id.includes('document_count'));
      
      // If ONLY metadata exists (no real projects), treat as empty database
      if (actualProjects.length === 0) {
        systemPrompt += `\n\n⚠️ DATABASE STATUS: The RECAP Project Hub database is currently EMPTY. There are NO research projects stored yet.

CRITICAL: You MUST inform the user that:
1. The database contains zero (0) research projects
2. You cannot answer questions about specific projects, abstracts, authors, or research details
3. They need to wait for an administrator to upload projects to the database

FORBIDDEN: Do NOT invent, fabricate, or make up any project data. Do NOT provide examples of "what projects might look like." Do NOT pretend projects exist.

ALLOWED: You may explain what this platform does and how it will work once projects are added.`;
      } else {
        // Database has real projects
        if (countProject && countProject.projects_document_count) {
          systemPrompt += `\n\n📊 DATABASE STATS:
- Total projects in database: ${countProject.projects_document_count}`;
        }
        
        systemPrompt += `\n\nDatabase Context (Relevant Research Projects from Pinecone):\n`;
        relevantProjects.forEach((project, idx) => {
          // Skip the count document in project listing
          if (project.id && project.id.includes('document_count')) return;
          
          systemPrompt += `\n--- Project ${idx + 1} ---`;
          systemPrompt += `\nTitle: ${project.title || 'Untitled'}`;
          if (project.authors) systemPrompt += `\nAuthors: ${project.authors}`;
          if (project.program) systemPrompt += `\nProgram: ${project.program}`;
          if (project.year) systemPrompt += `\nYear: ${project.year}`;
          if (project.adviser) systemPrompt += `\nAdviser: ${project.adviser}`;
          if (project.keywords) systemPrompt += `\nKeywords: ${Array.isArray(project.keywords) ? project.keywords.join(', ') : project.keywords}`;
          
          let abstractContent = project.abstract || project.fullText || '';
          if (!abstractContent && project.text) {
            const abstractMatch = project.text.match(/Abstract:\s*([^]*?)(?=(?:\s*(?:Keywords|Adviser|Authors|Program|Year|Key Findings|Topics):|\n\n|\*$|$))/i);
            if (abstractMatch && abstractMatch[1]) {
              abstractContent = abstractMatch[1].trim();
            }
          }
          if (abstractContent) {
            systemPrompt += `\nAbstract: ${abstractContent}`;
          } else if (project.text) {
            systemPrompt += `\nProject Summary: ${project.text}`;
          }
          if (project.score !== undefined) {
            systemPrompt += `\nRelevance: ${(project.score * 100).toFixed(1)}%`;
          }
          systemPrompt += `\n`;
        });
        
        systemPrompt += `\n\nPlease answer the user's inquiry accurately based on the project records and abstracts provided above.`;
      }
    } else {
      systemPrompt += `\n\n⚠️ DATABASE STATUS: The RECAP Project Hub database is currently EMPTY. There are NO research projects stored yet.

CRITICAL: You MUST inform the user that:
1. The database contains zero (0) research projects
2. You cannot answer questions about specific projects, abstracts, authors, or research details
3. They need to wait for an administrator to upload projects to the database

FORBIDDEN: Do NOT invent, fabricate, or make up any project data. Do NOT provide examples of "what projects might look like." Do NOT pretend projects exist.

ALLOWED: You may explain what this platform does and how it will work once projects are added.`;
    }
    
    // Build messages array
    const messages = [
      { role: 'system', content: systemPrompt },
      ...conversationHistory,
      { role: 'user', content: message }
    ];
    
    // If streaming requested (or by default)
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      if (typeof res.flushHeaders === 'function') {
        res.flushHeaders();
      }

      await streamAIWithFallback(messages, 1000, res, { projectsUsed: relevantProjects.length, preferredProvider });
    } else {
      // Non-streaming fallback
      const result = await callAIWithFallback(messages, 1000, preferredProvider);
      
      if (result.success) {
        res.json({
          success: true,
          response: result.response,
          provider: result.provider,
          providerKey: result.providerKey,
          attemptNumber: result.attemptNumber,
          projectsUsed: relevantProjects.length
        });
      } else {
        res.status(503).json({
          success: false,
          error: 'All AI providers unavailable',
          details: result.errors,
          fallbackMessage: result.message
        });
      }
    }
    
  } catch (error) {
    console.error('❌ Chat error:', error);
    if (!res.headersSent) {
      res.status(500).json({
        success: false,
        error: 'Chat failed',
        details: error.message
      });
    } else {
      res.write(`data: ${JSON.stringify({ type: 'error', error: error.message })}\n\n`);
      res.end();
    }
  }
});

/**
 * Individual provider endpoints (for testing/direct access with optional streaming)
 */
async function handleSingleProviderChat(providerKey, req, res) {
  try {
    const { messages, maxTokens = 1000, stream = false } = req.body;
    const provider = AI_PROVIDERS[providerKey];
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      if (typeof res.flushHeaders === 'function') res.flushHeaders();

      res.write(`data: ${JSON.stringify({ type: 'start', provider: provider.name, providerKey, attemptNumber: 1 })}\n\n`);
      
      let fullText = '';
      await streamFromAIProvider(providerKey, messages, maxTokens, (token) => {
        fullText += token;
        res.write(`data: ${JSON.stringify({ type: 'token', content: token })}\n\n`);
        if (typeof res.flush === 'function') res.flush();
      });

      res.write(`data: ${JSON.stringify({ type: 'done', fullResponse: fullText, provider: provider.name, providerKey, attemptNumber: 1 })}\n\n`);
      res.end();
    } else {
      const response = await callAIProvider(providerKey, messages, maxTokens);
      res.json({ success: true, response, provider: provider.name, providerKey });
    }
  } catch (error) {
    console.error(`❌ ${providerKey} chat error:`, error.message);
    if (!res.headersSent) {
      res.status(500).json({ success: false, error: error.message });
    } else {
      res.write(`data: ${JSON.stringify({ type: 'error', error: error.message })}\n\n`);
      res.end();
    }
  }
}

app.post('/api/chat/mistral', (req, res) => handleSingleProviderChat('mistral', req, res));
app.post('/api/chat/groq', (req, res) => handleSingleProviderChat('groq', req, res));
app.post('/api/chat/gemini', (req, res) => handleSingleProviderChat('gemini', req, res));
app.post('/api/chat/openrouter', (req, res) => handleSingleProviderChat('openrouter', req, res));

// ============================================
// STAFF COPILOT RAG ENDPOINT (ADMIN & LIBRARIAN)
// ============================================
/**
 * POST /api/staff-copilot/chat
 * Role-Based RAG Copilot for Head Librarian (Admin) and Library Staff (Librarian)
 * Vector Index: recaps-staff-copilot (multilingual-e5-large)
 */
app.post('/api/staff-copilot/chat', async (req, res) => {
  try {
    const { message, userRole = 'admin', history = [] } = req.body;

    if (!message || message.trim() === '') {
      return res.status(400).json({ error: 'Message is required' });
    }

    const role = (userRole === 'librarian') ? 'librarian' : 'admin';
    const staffCopilotHost = process.env.PINECONE_STAFF_COPILOT_HOST || 'recaps-staff-copilot-wf1edxk.svc.aped-4627-b74a.pinecone.io';

    console.log(`\n🤖 Staff Copilot Query [Role: ${role}]: "${message.substring(0, 60)}..."`);

    // Step 1: Embed query using Pinecone Inference API
    let retrievedChunks = [];
    try {
      const embedResponse = await fetch('https://api.pinecone.io/embed', {
        method: 'POST',
        headers: {
          'Api-Key': process.env.PINECONE_API_KEY,
          'Content-Type': 'application/json',
          'X-Pinecone-Api-Version': '2024-10'
        },
        body: JSON.stringify({
          model: 'multilingual-e5-large',
          parameters: {
            input_type: 'query',
            truncate: 'END'
          },
          inputs: [{ text: message }]
        })
      });

      if (embedResponse.ok) {
        const embedData = await embedResponse.json();
        const queryVector = embedData.data[0].values;

        // Step 2: Query Pinecone with Role-Based Metadata Filter
        const allowedScopes = (role === 'librarian')
          ? ['librarian', 'shared']
          : (message.toLowerCase().includes('librarian') ? ['admin', 'librarian', 'shared'] : ['admin', 'shared']);
        const roleFilter = { role_scope: { $in: allowedScopes } };

        const queryUrl = `https://${staffCopilotHost}/query`;
        const pineconeRes = await fetch(queryUrl, {
          method: 'POST',
          headers: {
            'Api-Key': process.env.PINECONE_API_KEY,
            'Content-Type': 'application/json',
            'X-Pinecone-Api-Version': '2025-04'
          },
          body: JSON.stringify({
            vector: queryVector,
            topK: 3,
            includeMetadata: true,
            filter: roleFilter
          })
        });

        if (pineconeRes.ok) {
          const queryResult = await pineconeRes.json();
          retrievedChunks = (queryResult.matches || []).map(m => ({
            id: m.id,
            score: m.score,
            title: m.metadata?.title || '',
            category: m.metadata?.category || '',
            landmark: m.metadata?.landmark || '',
            text: m.metadata?.text || ''
          }));
          console.log(`✓ Retrieved ${retrievedChunks.length} RAG guidance chunks from Pinecone (top score: ${retrievedChunks[0]?.score?.toFixed(3) || 0})`);
        } else {
          console.warn('⚠️ Pinecone staff copilot query failed:', await pineconeRes.text());
        }
      }
    } catch (ragErr) {
      console.warn('⚠️ Staff Copilot RAG retrieval error (falling back to baseline instructions):', ragErr.message);
    }

    // Step 3: Construct Grounded System Prompt
    const rolePersona = (role === 'librarian')
      ? `You are the RE-CAPS Librarian Cataloging Assistant for Cebu Technological University (CTU) Daanbantayan Campus Library.
Your role is to guide the library staff member on catalog intake, physical book shelf numbering, call numbers, accession codes, and drafting thesis submissions.
CRITICAL ROLE BOUNDARY: If the user asks how to approve, directly publish, or delete catalog records, explain politely and clearly that these administrative governance actions require evaluation and action by the Head Librarian / Administrator.`
      : `You are the RE-CAPS Executive Librarian AI Copilot for Cebu Technological University (CTU) Daanbantayan Campus Library.
Your role is to guide the Head Librarian & Administrator on complete operational governance: adding/uploading projects (+ Add Project), bulk importing spreadsheets, evaluating and approving librarian submissions, syncing Pinecone AI search, unlocking and deleting records, printing Dean/CHED accreditation reports, and maintenance mode.
CRITICAL SCOPE & ANTI-HALLUCINATION CONSTRAINTS:
1. The user is an ADMINISTRATOR (Head Librarian). They manage the entire system from the Admin Dashboard tabs.
2. If the user asks how to upload, add, or catalog a project/thesis, guide them through the Admin "+ Add Project" modal (#project-modal) in the Projects section. Note that Admin submissions are DIRECTLY PUBLISHED and do NOT require approval.
3. NEVER instruct the Admin to log in to an external "Librarian Portal" or submit proposals for approval.
4. NEVER hallucinate fake URLs or external web domains (e.g., "ctu-daanbantayan.library/portal"). RE-CAPS has no external portal URLs; all actions are performed strictly within the Admin Dashboard navigation tabs.`;

    let systemPrompt = `${rolePersona}

MANDATORY 3-TIER RESPONSE SYNTAX (Follow this exact structure for operational guidance):
1. SPATIAL LANDMARK: Start with the exact visual location on the screen, on a SINGLE LINE, e.g.:
   [Area: <Primary Section Name> > <Sub-Area/Table>] > [Element: <Button Name, Color & Icon>]

   PRIMARY SECTION NAMES FOR ADMIN DASHBOARD (Select strictly the one matching the operational domain):
   - Users Section: All user account management (deleting users, creating librarians, student/faculty accounts, password resets, unlocking accounts, role privileges).
   - Projects Section: Capstone thesis management (adding projects, cataloging shelf locations, Excel bulk import, Pinecone AI sync, editing/deleting projects).
   - Approvals Section: Evaluating and publishing librarian thesis submissions.
   - Analytics Section: Accreditation reports, printing official CHED reports, research intake charts, program breakdown.
   - Activity Section: Repository audit trails, security logs, history of changes.
   - Settings Section: Maintenance mode, registration toggle, system preferences.
   - Dashboard Overview: Top-level KPI counter cards.

CRITICAL ANTI-CONFUSION DIRECTIVES:
- If the user asks about ANY user account action (e.g., deleting a user, removing an account, creating a librarian, managing students), the primary area MUST be 'Users Section' (e.g., [Area: Users Section > Users Table]). NEVER route user account inquiries to Projects or capstone tables!
- If the user asks about capstone theses, manuscripts, or shelf locations, the primary area is 'Projects Section'.
- NEVER mix up user deletion (in Users Section) with project deletion (in Projects Section).

2. STEP-BY-STEP ACTION: Clear, concise numbered steps (1, 2, 3...) explaining what to click, input, or inspect.
3. PRECAUTION / POLICY ALERT: Start with ⚠️ Precaution: followed by any critical policy rule, security note, or data risk.

CRITICAL FORMATTING INSTRUCTIONS (STRICT):
- DO NOT output introductory conversational preambles (do NOT say "Here is your 3-TIER...", "Sure, here are the steps...", "Below is the guidance..."). Start IMMEDIATELY with line 1 (the SPATIAL LANDMARK).
- Keep [Area: ...] > [Element: ...] on a SINGLE continuous line without splitting brackets across lines.
- DO NOT output decorative horizontal rules (--- or ***).
- Number all actionable steps (1., 2., 3.).
- Close with ⚠️ Precaution: <risk or policy note>.

GROUNDING RULE: Base your guidance strictly on the verified Library Documentation Context below. NEVER hallucinate button locations, colors, or external URLs.`;

    if (retrievedChunks.length > 0) {
      systemPrompt += `\n\n--- VERIFIED LIBRARY DOCUMENTATION CONTEXT (PINECONE RAG) ---\n`;
      retrievedChunks.forEach((chunk, i) => {
        systemPrompt += `\n[Context Item ${i + 1}: ${chunk.title}]\nLandmark: ${chunk.landmark}\n${chunk.text}\n`;
      });
      systemPrompt += `\n-----------------------------------------------------------`;
    }

    // Build messages array
    const messages = [
      { role: 'system', content: systemPrompt }
    ];

    // Add recent history if provided
    if (Array.isArray(history) && history.length > 0) {
      history.slice(-4).forEach(h => {
        if (h.role && h.content) {
          messages.push({ role: h.role, content: h.content });
        }
      });
    }

    messages.push({ role: 'user', content: message });

    // Step 4: Call cascading AI fallback engine
    const aiResult = await callAIWithFallback(messages, 800);

    if (aiResult.success) {
      return res.json({
        success: true,
        reply: aiResult.response,
        provider: aiResult.provider,
        role,
        ragRetrieved: retrievedChunks.length > 0,
        sources: retrievedChunks.map(c => ({ title: c.title, landmark: c.landmark, score: c.score }))
      });
    } else {
      return res.status(500).json({
        success: false,
        error: aiResult.message || 'All AI providers are currently unavailable'
      });
    }

  } catch (error) {
    console.error('❌ Staff Copilot endpoint error:', error);
    return res.status(500).json({ success: false, error: error.message });
  }
});

// ============================================
// CLOUDINARY IMAGE DELETION
// ============================================
/**
 * DELETE /api/cloudinary/delete
 * Delete image from Cloudinary
 */
app.delete('/api/cloudinary/delete', async (req, res) => {
  try {
    const { publicId } = req.body;
    
    if (!publicId) {
      return res.status(400).json({
        success: false,
        message: 'Public ID is required'
      });
    }

    if (!cloudinary) {
      return res.status(503).json({
        success: false,
        message: 'Cloudinary module not configured on server'
      });
    }

    console.log(`🗑️  Deleting image from Cloudinary: ${publicId}`);

    const result = await cloudinary.uploader.destroy(publicId);

    if (result.result === 'ok') {
      console.log(`✓ Image deleted successfully: ${publicId}`);
      return res.json({
        success: true,
        message: 'Image deleted successfully',
        result: result.result
      });
    } else if (result.result === 'not found') {
      console.warn(`⚠️  Image not found in Cloudinary: ${publicId}`);
      return res.json({
        success: true, // Not an error, just already deleted
        message: 'Image not found (may have been deleted already)',
        result: result.result
      });
    } else {
      console.error(`❌ Cloudinary deletion failed: ${result.result}`);
      return res.status(500).json({
        success: false,
        message: `Deletion failed: ${result.result}`
      });
    }
  } catch (error) {
    console.error('❌ Error deleting image from Cloudinary:', error);
    return res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// ============================================
// USER MANAGEMENT ENDPOINTS
// ============================================

/**
 * Helper function to handle user deletion from Firebase Auth and Firestore
 */
async function handleUserDeletion(uid, email, res) {
  if (!uid && !email) {
    return res.status(400).json({
      success: false,
      message: 'User UID or email is required'
    });
  }

  console.log(`\n🗑️  [USER DELETE] Request to delete user: UID=${uid || 'N/A'}, Email=${email || 'N/A'}`);

  let authDeleted = false;
  let targetUid = uid;
  let authMessage = '';

  // 1. Try deleting from Firebase Authentication by UID
  if (targetUid) {
    try {
      await admin.auth().deleteUser(targetUid);
      authDeleted = true;
      authMessage = `Deleted user ${targetUid} from Firebase Authentication`;
      console.log(`✓ ${authMessage}`);
    } catch (authError) {
      if (authError.code === 'auth/user-not-found') {
        console.log(`ℹ️ User UID ${targetUid} not found in Firebase Authentication (already deleted)`);
        authDeleted = true;
        authMessage = 'User already absent from Firebase Authentication';
      } else {
        console.warn(`⚠️ Firebase Auth delete by UID failed: ${authError.message}`);
      }
    }
  }

  // 2. Fallback: If not deleted and email is provided, lookup by email and delete
  if (!authDeleted && email) {
    try {
      const userRecord = await admin.auth().getUserByEmail(email);
      targetUid = userRecord.uid;
      await admin.auth().deleteUser(userRecord.uid);
      authDeleted = true;
      authMessage = `Deleted user ${userRecord.uid} (${email}) from Firebase Authentication via email lookup`;
      console.log(`✓ ${authMessage}`);
    } catch (emailError) {
      if (emailError.code === 'auth/user-not-found') {
        console.log(`ℹ️ User with email ${email} not found in Firebase Authentication`);
        authDeleted = true;
        authMessage = 'User already absent from Firebase Authentication';
      } else {
        console.error(`❌ Firebase Auth delete by email lookup failed: ${emailError.message}`);
        authMessage = emailError.message;
      }
    }
  }

  // 3. Delete user document from Firestore
  let firestoreDeleted = false;
  if (targetUid) {
    try {
      const userDocRef = db.collection('users').doc(targetUid);
      const doc = await userDocRef.get();
      if (doc.exists) {
        await userDocRef.delete();
        firestoreDeleted = true;
        console.log(`✓ Deleted Firestore document for user ${targetUid}`);
      } else {
        firestoreDeleted = true;
        console.log(`ℹ️ Firestore document for user ${targetUid} already absent`);
      }
    } catch (fsError) {
      console.warn(`⚠️ Firestore document deletion warning: ${fsError.message}`);
    }
  }

  return res.json({
    success: true,
    message: 'User deletion processed successfully',
    authDeleted,
    firestoreDeleted,
    uid: targetUid,
    details: authMessage
  });
}

/**
 * DELETE /api/users/:uid
 * Delete user from Firebase Authentication and Firestore
 */
app.delete('/api/users/:uid', async (req, res) => {
  try {
    const { uid } = req.params;
    const { email } = req.body || {};
    return await handleUserDeletion(uid, email, res);
  } catch (err) {
    console.error('❌ Error in DELETE /api/users/:uid:', err);
    return res.status(500).json({ success: false, message: err.message });
  }
});

/**
 * POST /api/users/delete
 * Fallback POST endpoint for deleting user
 */
app.post('/api/users/delete', async (req, res) => {
  try {
    const { uid, userId, email } = req.body || {};
    const targetUid = uid || userId;
    return await handleUserDeletion(targetUid, email, res);
  } catch (err) {
    console.error('❌ Error in POST /api/users/delete:', err);
    return res.status(500).json({ success: false, message: err.message });
  }
});

/**
 * POST /api/users/create-librarian
 * Create librarian user in Firebase Authentication & Firestore via Admin SDK
 * Never affects or resets the active admin session on the frontend
 */
app.post('/api/users/create-librarian', async (req, res) => {
  try {
    // 0. Authenticate caller: require a valid Firebase ID token belonging to an admin
    const authHeader = req.headers.authorization || '';
    const idToken = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
    if (!idToken) {
      return res.status(401).json({ success: false, message: 'Missing authentication token' });
    }

    let callerUid;
    try {
      const decoded = await admin.auth().verifyIdToken(idToken);
      callerUid = decoded.uid;
    } catch (tokenErr) {
      return res.status(401).json({ success: false, message: 'Invalid or expired authentication token' });
    }

    const callerDoc = await db.collection('users').doc(callerUid).get();
    if (!callerDoc.exists || callerDoc.data().userType !== 'admin') {
      return res.status(403).json({ success: false, message: 'Only administrators can create librarian accounts' });
    }

    const { email, password, fullName } = req.body || {};
    const adminUid = callerUid;

    if (!email || !password || !fullName) {
      return res.status(400).json({
        success: false,
        message: 'Full name, email, and password are required'
      });
    }

    const trimmedEmail = email.trim().toLowerCase();
    const trimmedName = fullName.trim();

    if (password.length < 6) {
      return res.status(400).json({
        success: false,
        message: 'Password must be at least 6 characters'
      });
    }

    // 1. Create user in Firebase Authentication
    let userRecord;
    try {
      userRecord = await admin.auth().createUser({
        email: trimmedEmail,
        password: password,
        displayName: trimmedName
      });
      console.log(`✓ Created Firebase Auth user: ${userRecord.uid} (${trimmedEmail})`);
    } catch (authErr) {
      console.error('❌ Firebase Auth createUser error:', authErr);
      if (authErr.code === 'auth/email-already-exists') {
        return res.status(409).json({
          success: false,
          code: authErr.code,
          message: 'An account with this email address already exists'
        });
      }
      if (authErr.code === 'auth/invalid-email') {
        return res.status(400).json({
          success: false,
          code: authErr.code,
          message: 'The email address is invalid'
        });
      }
      return res.status(400).json({
        success: false,
        code: authErr.code,
        message: authErr.message || 'Failed to create user in authentication system'
      });
    }

    // 2. Set custom user claims for role (optional but best practice for security rules)
    try {
      await admin.auth().setCustomUserClaims(userRecord.uid, { role: 'librarian' });
    } catch (claimErr) {
      console.warn('⚠️ Could not set custom claims:', claimErr.message);
    }

    // 3. Create Firestore document in 'users' collection
    try {
      await db.collection('users').doc(userRecord.uid).set({
        email: trimmedEmail,
        fullName: trimmedName,
        userType: 'librarian',
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
        createdBy: adminUid || 'admin',
        lastLogin: null,
        photoURL: null
      });
      console.log(`✓ Created Firestore profile for librarian: ${userRecord.uid}`);
    } catch (fsErr) {
      console.error('❌ Firestore write error for new librarian:', fsErr);
      // Clean up auth user if firestore write failed
      try {
        await admin.auth().deleteUser(userRecord.uid);
      } catch (cleanupErr) {
        console.error('❌ Failed to cleanup auth user after firestore error:', cleanupErr);
      }
      return res.status(500).json({
        success: false,
        message: 'Failed to create librarian record in database'
      });
    }

    return res.status(201).json({
      success: true,
      message: 'Librarian account created successfully',
      user: {
        uid: userRecord.uid,
        email: trimmedEmail,
        fullName: trimmedName,
        userType: 'librarian'
      }
    });
  } catch (err) {
    console.error('❌ Error in /api/users/create-librarian:', err);
    return res.status(500).json({
      success: false,
      message: err.message || 'Internal server error while creating librarian'
    });
  }
});

// ============================================
// HEALTH CHECK
// ============================================
app.get('/api/health', (req, res) => {
  const providersStatus = {};
  
  for (const [key, provider] of Object.entries(AI_PROVIDERS)) {
    providersStatus[key] = {
      name: provider.name,
      configured: !!provider.apiKey,
      model: provider.model
    };
  }
  
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    firebase: 'connected',
    pinecone: 'connected',
    aiProviders: providersStatus
  });
});

// ============================================
// FEEDBACK ENDPOINT
// ============================================

// ============================================
// START SERVER
// ============================================
app.listen(PORT, () => {
  console.log(`\n${'='.repeat(50)}`);
  console.log(`🚀 RECAP Backend Server`);
  console.log(`${'='.repeat(50)}`);
  console.log(`✓ Server running on port ${PORT}`);
  console.log(`✓ Firebase: Connected`);
  console.log(`✓ Pinecone: Connected (${indexName})`);
  console.log(`\n📡 AI Providers (Priority Order):`);
  console.log(`    1. ${AI_PROVIDERS.mistral.name} ${AI_PROVIDERS.mistral.apiKey ? '✓' : '✗'}`);
  console.log(`    2. ${AI_PROVIDERS.groq.name} ${AI_PROVIDERS.groq.apiKey ? '✓' : '✗'}`);
  console.log(`    3. ${AI_PROVIDERS.gemini.name} ${AI_PROVIDERS.gemini.apiKey ? '✓' : '✗'}`);
  console.log(`    4. ${AI_PROVIDERS.openrouter.name} ${AI_PROVIDERS.openrouter.apiKey ? '✓' : '✗'}`);
  console.log(`\n🌐 Endpoints:`);
  console.log(`    POST   /api/pinecone/search - RAG semantic search`);
  console.log(`    POST   /api/chat - AI chat with fallback`);
  console.log(`    POST   /api/feedback - Send feedback email`);
  console.log(`    DELETE /api/users/:uid - Delete user from Auth & Firestore`);
  console.log(`    POST   /api/users/delete - Delete user fallback`);
  console.log(`    GET    /api/health - Health check`);
  console.log(`${'='.repeat(50)}\n`);
});

// ============================================
// ERROR HANDLING
// ============================================
process.on('unhandledRejection', (error) => {
  console.error('❌ Unhandled rejection:', error);
});

process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught exception:', error);
  process.exit(1);
});

// ============================================
// FEEDBACK ENDPOINT
// ============================================

/**
 * POST /api/feedback
 * Send feedback email via Resend API
 */
app.post('/api/feedback', async (req, res) => {
  try {
    const { name, email, feeling, subject, message, images } = req.body;
    
    console.log('📧 Processing feedback from:', email);
    
    // Validation
    if (!name || !email || !message) {
      return res.status(400).json({
        success: false,
        error: 'Name, email, and message are required'
      });
    }
    
    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid email address'
      });
    }
    
    // 1. Always store feedback in Firestore first so no feedback is ever lost
    try {
      await db.collection('feedback').add({
        name,
        email,
        subject: subject || 'General Feedback',
        message,
        status: 'unread',
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
        ipAddress: req.ip || 'unknown',
        userAgent: req.headers['user-agent'] || 'unknown'
      });
      console.log('✓ Feedback stored in Firestore');
    } catch (dbError) {
      console.warn('⚠️ Failed to store in Firestore:', dbError.message);
    }

    // 2. Resend configuration with trimming to prevent whitespace/newline issues
    const activeApiKey = (process.env.RESEND_API_KEY || RESEND_API_KEY || '').trim();
    const activeRecipientEmail = (process.env.FEEDBACK_RECIPIENT_EMAIL || FEEDBACK_RECIPIENT_EMAIL || '').trim();
    const activeFromEmail = (process.env.RESEND_FROM_EMAIL || RESEND_FROM_EMAIL || 'onboarding@resend.dev').trim();
    const activeFromName = (process.env.RESEND_FROM_NAME || RESEND_FROM_NAME || 'RECAP Support').trim();

    // Check if Resend is configured - if not, feedback is still saved in Firestore
    if (!activeApiKey) {
      console.warn('⚠️ Resend API not configured. Feedback saved in database only.');
      return res.json({
        success: true,
        message: 'Feedback received successfully!'
      });
    }

    const feedbackSubject = subject || 'New Feedback - RECAP Project Hub';
    const recipientEmail = activeRecipientEmail;
    
    // Map feeling to emoji
    const feelingEmojis = {
      'excited': '🤩',
      'happy': '😊',
      'neutral': '😐',
      'confused': '😕',
      'sad': '😢',
      'angry': '😠'
    };
    const feelingEmoji = feelingEmojis[feeling] || '💬';
    const feelingText = feeling ? feeling.charAt(0).toUpperCase() + feeling.slice(1) : 'Not specified';
    
    // Primary color from system
    const PRIMARY_COLOR = '#08D488';
    
    // HTML email template for admin
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
          }
          .email-container {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
          }
          .header {
            background: ${PRIMARY_COLOR};
            color: white;
            padding: 30px;
            text-align: center;
          }
          .header h1 {
            margin: 0 0 10px 0;
            font-size: 28px;
            font-weight: 700;
          }
          .header p {
            margin: 0;
            opacity: 0.95;
            font-size: 14px;
          }
          .content {
            padding: 30px;
          }
          .field {
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e9ecef;
          }
          .field:last-child {
            border-bottom: none;
          }
          .label {
            font-weight: 600;
            color: ${PRIMARY_COLOR};
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
          }
          .value {
            background: #f8f9fa;
            padding: 12px 15px;
            border-radius: 8px;
            border-left: 3px solid ${PRIMARY_COLOR};
            font-size: 15px;
          }
          .value a {
            color: ${PRIMARY_COLOR};
            text-decoration: none;
          }
          .message-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid ${PRIMARY_COLOR};
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 15px;
            line-height: 1.7;
          }
          .feeling-badge {
            display: inline-block;
            background: ${PRIMARY_COLOR};
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
          }
          .images-section {
            margin-top: 20px;
          }
          .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 10px;
          }
          .image-item {
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid #e9ecef;
          }
          .image-item img {
            width: 100%;
            height: 150px;
            object-fit: cover;
            display: block;
          }
          .footer {
            padding: 20px 30px;
            background: #f8f9fa;
            text-align: center;
            font-size: 13px;
            color: #6c757d;
          }
        </style>
      </head>
      <body>
        <div class="email-container">
          <div class="header">
            <h1>📬 New Feedback Received</h1>
            <p>RECAP Project Hub</p>
          </div>
          <div class="content">
            <div class="field">
              <div class="label">From</div>
              <div class="value">${name}</div>
            </div>
            <div class="field">
              <div class="label">Email</div>
              <div class="value"><a href="mailto:${email}">${email}</a></div>
            </div>
            <div class="field">
              <div class="label">Feeling</div>
              <div class="value">
                <span class="feeling-badge">${feelingEmoji} ${feelingText}</span>
              </div>
            </div>
            ${subject ? `<div class="field"><div class="label">Subject</div><div class="value">${subject}</div></div>` : ''}
            <div class="field">
              <div class="label">Message</div>
              <div class="message-box">${message}</div>
            </div>
            ${images && images.length > 0 ? `
              <div class="images-section">
                <div class="label">Attached Images (${images.length})</div>
                <div class="image-grid">
                  ${images.map((url, idx) => `
                    <div class="image-item">
                      <a href="${url}" target="_blank">
                        <img src="${url}" alt="Attachment ${idx + 1}" />
                      </a>
                    </div>
                  `).join('')}
                </div>
              </div>
            ` : ''}
          </div>
          <div class="footer">
            <p>Sent from RECAP Project Hub Feedback System</p>
          </div>
        </div>
      </body>
      </html>
    `;
    
    // 3. Send admin notification via Resend API (OPTIONAL - only if FEEDBACK_RECIPIENT_EMAIL is configured)
    if (recipientEmail) {
      console.log(`📤 Sending admin notification email to ${recipientEmail}...`);
      try {
        const fromField = activeFromName ? `${activeFromName} <${activeFromEmail}>` : activeFromEmail;
        
        const adminEmailResponse = await fetch('https://api.resend.com/emails', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${activeApiKey}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            from: fromField,
            to: recipientEmail,
            reply_to: email,
            subject: feedbackSubject,
            html: htmlContent
          })
        });
        
        if (!adminEmailResponse.ok) {
          const errorData = await adminEmailResponse.json().catch(() => ({}));
          console.warn('⚠️ Admin email failed:', errorData.message || adminEmailResponse.statusText);
        } else {
          console.log('✓ Feedback admin email sent successfully to', recipientEmail);
        }
      } catch (emailError) {
        console.warn('⚠️ Failed to send admin email:', emailError.message);
      }
    } else {
      console.log('ℹ️ FEEDBACK_RECIPIENT_EMAIL not set — skipping admin email notification (saving 1 email quota per feedback)');
    }
    
    // Send user confirmation
    try {
      const userEmailContent = `
        <!DOCTYPE html>
        <html>
        <head>
          <style>
            body {
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
              line-height: 1.6;
              color: #333;
              max-width: 600px;
              margin: 0 auto;
              padding: 20px;
              background: #f5f5f5;
            }
            .email-container {
              background: white;
              border-radius: 12px;
              overflow: hidden;
              box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .header {
              background: ${PRIMARY_COLOR};
              color: white;
              padding: 40px 30px;
              text-align: center;
            }
            .checkmark-wrapper {
              display: inline-block;
              width: 70px;
              height: 70px;
              background: rgba(255,255,255,0.2);
              border-radius: 50%;
              padding: 15px;
              margin-bottom: 15px;
            }
            .checkmark {
              width: 40px;
              height: 40px;
              stroke: white;
              stroke-width: 4;
              fill: none;
            }
            .header h1 {
              margin: 0;
              font-size: 28px;
              font-weight: 700;
            }
            .content {
              padding: 35px 30px;
              text-align: center;
            }
            .greeting {
              font-size: 18px;
              color: #333;
              margin-bottom: 10px;
            }
            .greeting strong {
              color: ${PRIMARY_COLOR};
            }
            .message-preview {
              background: #f8f9fa;
              padding: 20px;
              border-radius: 8px;
              margin: 25px 0;
              border-left: 4px solid ${PRIMARY_COLOR};
              text-align: left;
            }
            .message-preview p {
              margin: 0;
              color: #6c757d;
              font-style: italic;
              line-height: 1.6;
            }
            .info-box {
              background: rgba(8,212,136,0.08);
              border: 1px solid ${PRIMARY_COLOR};
              border-radius: 8px;
              padding: 20px;
              margin: 25px 0;
            }
            .info-item {
              display: flex;
              align-items: center;
              justify-content: center;
              gap: 10px;
              margin: 10px 0;
              color: #333;
              font-size: 15px;
            }
            .info-icon {
              color: ${PRIMARY_COLOR};
              font-size: 20px;
            }
            .footer {
              padding: 25px 30px;
              background: #f8f9fa;
              text-align: center;
            }
            .footer p {
              margin: 5px 0;
              font-size: 14px;
              color: #6c757d;
            }
            .team-signature {
              color: ${PRIMARY_COLOR};
              font-weight: 600;
            }
          </style>
        </head>
        <body>
          <div class="email-container">
            <div class="header">
              <div class="checkmark-wrapper">
                <svg class="checkmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52">
                  <circle cx="26" cy="26" r="25"/>
                  <path d="M14.1 27.2l7.1 7.2 16.7-16.8" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </div>
              <h1>Thank You!</h1>
            </div>
            <div class="content">
              <p class="greeting">Hi <strong>${name}</strong>! 👋</p>
              <p style="color: #6c757d; margin-bottom: 20px;">
                We've successfully received your feedback.
              </p>
              
              <div class="message-preview">
                <p>"${message.substring(0, 200)}${message.length > 200 ? '...' : ''}"</p>
              </div>
              
              <div class="info-box">
                <div class="info-item">
                  <span class="info-icon">✓</span>
                  <span>Your message has been sent to our team</span>
                </div>
                <div class="info-item">
                  <span class="info-icon">📧</span>
                  <span>We'll review and respond shortly</span>
                </div>
                ${feeling ? `
                  <div class="info-item">
                    <span class="info-icon">${feelingEmoji}</span>
                    <span>You felt: ${feelingText}</span>
                  </div>
                ` : ''}
              </div>
              
              <p style="color: #6c757d; font-size: 14px;">
                If you have any urgent concerns, please reply to this email directly.
              </p>
            </div>
            <div class="footer">
              <p class="team-signature">— The RECAP Team</p>
              <p>CTU Daanbantayan Campus • BSIT Program</p>
            </div>
          </div>
        </body>
        </html>
      `;
      
      const fromField = activeFromName ? `${activeFromName} <${activeFromEmail}>` : activeFromEmail;
      
      const userEmailResponse = await fetch('https://api.resend.com/emails', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${activeApiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          from: fromField,
          to: email,
          subject: 'Thank You for Your Feedback - RECAP',
          html: userEmailContent
        })
      });
      
      if (!userEmailResponse.ok) {
        const errorData = await userEmailResponse.json();
        console.warn('⚠️ User confirmation email failed:', errorData.message);
      } else {
        console.log('✓ Confirmation email sent to', email);
      }
    } catch (confirmError) {
      console.warn('⚠️ Failed to send confirmation email:', confirmError.message);
    }
    

    
    res.json({
      success: true,
      message: 'Feedback sent successfully! Check your email for confirmation.'
    });
    
  } catch (error) {
    console.error('❌ Feedback error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to send feedback. Please try again later.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});
