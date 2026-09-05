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
const nodemailer = require('nodemailer');
const fs = require('fs');

// Polyfill fetch for Node.js < 18
if (typeof fetch === 'undefined') {
  global.fetch = require('node-fetch');
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
// NODEMAILER CONFIGURATION
// ============================================
let transporter;

try {
  transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST || 'smtp.gmail.com',
    port: parseInt(process.env.SMTP_PORT) || 587,
    secure: process.env.SMTP_SECURE === 'true', // true for 465, false for other ports
    auth: {
      user: process.env.SMTP_USER,
      pass: process.env.SMTP_PASS
    }
  });

  // Verify SMTP connection
  if (process.env.SMTP_USER && process.env.SMTP_PASS) {
    transporter.verify((error, success) => {
      if (error) {
        console.warn('⚠️  SMTP verification failed:', error.message);
        console.warn('   Feedback emails may not work. Check your SMTP credentials.');
      } else {
        console.log('✓ SMTP transporter ready');
      }
    });
  } else {
    console.warn('⚠️  SMTP credentials not configured. Feedback emails will not be sent.');
  }
} catch (error) {
  console.error('❌ Failed to initialize email transporter:', error.message);
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
    endpoint: (apiKey) => `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=${apiKey}`,
    streamEndpoint: (apiKey) => `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:streamGenerateContent?alt=sse&key=${apiKey}`,
    model: 'gemini-flash-latest',
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
    model: 'google/gemma-4-26b-a4b-it:free',
    apiKey: process.env.OPENROUTER_API_KEY,
    headers: (apiKey) => ({
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://recaps-project-hub.onrender.com',
      'X-Title': 'RECAP Project Hub'
    }),
    formatRequest: (messages, maxTokens, stream = false) => ({
      model: 'google/gemma-4-26b-a4b-it:free',
      messages,
      max_tokens: maxTokens,
      temperature: 0.7,
      stream: !!stream
    }),
    extractResponse: (data) => data.choices[0].message.content,
    extractStreamDelta: (json) => json.choices?.[0]?.delta?.content || ''
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
 * 4-Tier cascading fallback system with real-time SSE streaming
 */
async function streamAIWithFallback(messages, maxTokens = 1000, res, metadata = {}) {
  const providers = ['mistral', 'groq', 'gemini', 'openrouter'];
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
async function callAIWithFallback(messages, maxTokens = 1000) {
  const providers = ['mistral', 'groq', 'gemini', 'openrouter'];
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
 * Upsert a project to Pinecone
 */
async function upsertProjectToPinecone(projectId, projectData) {
  try {
    console.log(`📤 Upserting project to Pinecone: ${projectId}`);
    
    // Build searchable text combining key fields
    const searchableText = [
      projectData.title || '',
      Array.isArray(projectData.authors) ? projectData.authors.join(' ') : (projectData.authors || ''),
      projectData.abstract || '',
      projectData.keyFindings || '',
      projectData.program || '',
      projectData.adviser || '',
      Array.isArray(projectData.topics) ? projectData.topics.join(' ') : '',
      Array.isArray(projectData.keywords) ? projectData.keywords.join(' ') : ''
    ].filter(Boolean).join(' ').trim();
    
    if (!searchableText) {
      console.warn('⚠️  No searchable content for project:', projectId);
      return;
    }
    
    // Generate embedding
    const embedding = await generateEmbedding(searchableText);
    
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
      text: searchableText,
      status: projectData.status || 'Completed'
    };
    
    // Upsert to Pinecone
    const upsertUrl = `https://${process.env.PINECONE_INDEX_HOST}/vectors/upsert`;
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
      })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Pinecone upsert failed: ${errorText}`);
    }
    
    console.log(`✓ Project upserted to Pinecone: ${projectId}`);
    return true;
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
  try {
    const { projectId, projectData } = req.body;
    
    if (!projectId || !projectData) {
      return res.status(400).json({ 
        success: false, 
        error: 'projectId and projectData are required' 
      });
    }
    
    await upsertProjectToPinecone(projectId, projectData);
    
    res.json({
      success: true,
      message: 'Project synced to Pinecone successfully',
      projectId
    });
  } catch (error) {
    console.error('❌ Sync error:', error);
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
      } catch (error) {
        results.failed++;
        results.errors.push({
          projectId: project.id,
          error: error.message
        });
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
    if (filter.year) {
      // Handle both single value and array of values
      if (Array.isArray(filter.year)) {
        metadataFilter.year = { $in: filter.year };
      } else {
        metadataFilter.year = { $eq: filter.year };
      }
    }
    if (filter.program) {
      // Handle both single value and array of values
      if (Array.isArray(filter.program)) {
        metadataFilter.program = { $in: filter.program };
      } else {
        metadataFilter.program = { $eq: filter.program };
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
    const { message, conversationHistory = [], relevantProjects = [], stream = true } = req.body;
    
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

10. Citation Format:
    - Always mention project title when referencing data
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
          
          const abstractContent = project.abstract || project.fullText || project.text || '';
          if (abstractContent) {
            systemPrompt += `\nAbstract: ${abstractContent}`;
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

      await streamAIWithFallback(messages, 1000, res, { projectsUsed: relevantProjects.length });
    } else {
      // Non-streaming fallback
      const result = await callAIWithFallback(messages, 1000);
      
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

/**
 * POST /api/feedback
 * Send feedback email via Nodemailer
 * 
 * Request body:
 * {
 *   name: string (required)
 *   email: string (required)
 *   subject: string (optional)
 *   message: string (required)
 * }
 */
app.post('/api/feedback', async (req, res) => {
  try {
    const { name, email, subject, message } = req.body;
    
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
    
    // Check if SMTP is configured
    if (!process.env.SMTP_USER || !process.env.SMTP_PASS) {
      console.error('❌ SMTP not configured. Cannot send feedback email.');
      return res.status(503).json({
        success: false,
        error: 'Email service is not configured. Please contact the administrator.'
      });
    }
    
    console.log(`📧 Processing feedback from: ${email}`);
    
    // Prepare email content
    const feedbackSubject = subject || 'New Feedback - RECAP Project Hub';
    const recipientEmail = process.env.FEEDBACK_RECIPIENT_EMAIL || process.env.SMTP_USER;
    
    // HTML email template
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
          }
          .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px 10px 0 0;
            text-align: center;
          }
          .header h1 {
            margin: 0;
            font-size: 24px;
          }
          .content {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 0 0 10px 10px;
          }
          .field {
            margin-bottom: 20px;
          }
          .label {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 5px;
          }
          .value {
            background: white;
            padding: 12px;
            border-radius: 5px;
            border-left: 3px solid #667eea;
          }
          .message-box {
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #764ba2;
            white-space: pre-wrap;
            word-wrap: break-word;
          }
          .footer {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #e9ecef;
            text-align: center;
            color: #6c757d;
            font-size: 12px;
          }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>📬 New Feedback Received</h1>
          <p style="margin: 5px 0 0 0;">RECAP Project Hub</p>
        </div>
        <div class="content">
          <div class="field">
            <div class="label">👤 From:</div>
            <div class="value">${name}</div>
          </div>
          
          <div class="field">
            <div class="label">📧 Email:</div>
            <div class="value"><a href="mailto:${email}" style="color: #667eea; text-decoration: none;">${email}</a></div>
          </div>
          
          ${subject ? `
          <div class="field">
            <div class="label">📋 Subject:</div>
            <div class="value">${subject}</div>
          </div>
          ` : ''}
          
          <div class="field">
            <div class="label">💬 Message:</div>
            <div class="message-box">${message}</div>
          </div>
          
          <div class="footer">
            <p>Received at ${new Date().toLocaleString('en-US', { timeZone: 'Asia/Manila', dateStyle: 'full', timeStyle: 'long' })}</p>
            <p>RECAP - Research Exploration & Capstone Archive Project</p>
          </div>
        </div>
      </body>
      </html>
    `;
    
    // Plain text version
    const textContent = `
New Feedback - RECAP Project Hub
================================

From: ${name}
Email: ${email}
${subject ? `Subject: ${subject}\n` : ''}
Message:
${message}

---
Received at ${new Date().toLocaleString('en-US', { timeZone: 'Asia/Manila', dateStyle: 'full', timeStyle: 'long' })}
RECAP - Research Exploration & Capstone Archive Project
    `.trim();
    
    // Send email to admin
    const mailOptions = {
      from: `"RECAP Feedback" <${process.env.SMTP_USER}>`,
      to: recipientEmail,
      replyTo: email,
      subject: feedbackSubject,
      text: textContent,
      html: htmlContent
    };
    
    await transporter.sendMail(mailOptions);
    console.log(`✓ Feedback email sent successfully to ${recipientEmail}`);
    
    // Send confirmation email to user
    const confirmationHtml = `
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
          }
          .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px 10px 0 0;
            text-align: center;
          }
          .content {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 0 0 10px 10px;
          }
          .success-icon {
            font-size: 48px;
            text-align: center;
            margin-bottom: 20px;
          }
          .button {
            display: inline-block;
            padding: 12px 30px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 20px;
          }
          .footer {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #e9ecef;
            text-align: center;
            color: #6c757d;
            font-size: 12px;
          }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>✅ Feedback Received</h1>
        </div>
        <div class="content">
          <div class="success-icon">🎉</div>
          <h2 style="color: #667eea; text-align: center;">Thank You, ${name}!</h2>
          <p>We've received your feedback and appreciate you taking the time to reach out to us.</p>
          <p>Our team will review your message and get back to you as soon as possible.</p>
          
          <div style="background: white; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <p style="margin: 0;"><strong>Your message:</strong></p>
            <p style="margin: 10px 0 0 0; color: #6c757d;">"${message.substring(0, 150)}${message.length > 150 ? '...' : ''}"</p>
          </div>
          
          <div style="text-align: center;">
            <a href="https://recaps-project-hub.onrender.com" class="button">Back to RECAP</a>
          </div>
          
          <div class="footer">
            <p>RECAP - Research Exploration & Capstone Archive Project</p>
            <p>CTU Daanbantayan Campus · BSIT Program</p>
          </div>
        </div>
      </body>
      </html>
    `;
    
    const confirmationText = `
Thank You for Your Feedback!

Hi ${name},

We've received your feedback and appreciate you taking the time to reach out to us.

Your message:
"${message.substring(0, 150)}${message.length > 150 ? '...' : ''}"

Our team will review your message and get back to you as soon as possible.

---
RECAP - Research Exploration & Capstone Archive Project
CTU Daanbantayan Campus · BSIT Program
    `.trim();
    
    try {
      const confirmationMailOptions = {
        from: `"RECAP Team" <${process.env.SMTP_USER}>`,
        to: email,
        subject: 'Thank You for Your Feedback - RECAP',
        text: confirmationText,
        html: confirmationHtml
      };
      
      await transporter.sendMail(confirmationMailOptions);
      console.log(`✓ Confirmation email sent to ${email}`);
    } catch (confirmError) {
      console.warn('⚠️  Failed to send confirmation email:', confirmError.message);
      // Don't fail the request if confirmation email fails
    }
    
    // Optional: Store feedback in Firestore
    try {
      await db.collection('feedback').add({
        name,
        email,
        subject: subject || 'General Feedback',
        message,
        status: 'unread',
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
        ipAddress: req.ip || req.headers['x-forwarded-for'] || 'unknown',
        userAgent: req.headers['user-agent'] || 'unknown'
      });
      console.log('✓ Feedback stored in Firestore');
    } catch (dbError) {
      console.warn('⚠️  Failed to store feedback in Firestore:', dbError.message);
      // Don't fail the request if database storage fails
    }
    
    res.json({
      success: true,
      message: 'Feedback sent successfully! Check your email for confirmation.'
    });
    
  } catch (error) {
    console.error('❌ Feedback submission error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to send feedback. Please try again later.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

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
  console.log(`    POST /api/pinecone/search - RAG semantic search`);
  console.log(`    POST /api/chat - AI chat with fallback`);
  console.log(`    POST /api/feedback - Send feedback email`);
  console.log(`    GET  /api/health - Health check`);
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
 * Send feedback email via Nodemailer
 */
app.post('/api/feedback', async (req, res) => {
  try {
    const { name, email, subject, message } = req.body;
    
    console.log('📧 Feedback request received:', { name, email, subject: subject || 'No subject' });
    
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
    
    // Check if SMTP is configured
    if (!process.env.SMTP_USER || !process.env.SMTP_PASS) {
      console.error('❌ SMTP not configured');
      return res.status(503).json({
        success: false,
        error: 'Email service is not configured. Please contact the administrator.'
      });
    }
    
    console.log('✓ Validation passed, preparing emails...');
    
    const feedbackSubject = subject || 'New Feedback - RECAP Project Hub';
    const recipientEmail = process.env.FEEDBACK_RECIPIENT_EMAIL || process.env.SMTP_USER;
    
    // HTML email template for admin
    const htmlContent = `
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
          }
          .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px 10px 0 0;
            text-align: center;
          }
          .content {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 0 0 10px 10px;
          }
          .field {
            margin-bottom: 20px;
          }
          .label {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 5px;
          }
          .value {
            background: white;
            padding: 12px;
            border-radius: 5px;
            border-left: 3px solid #667eea;
          }
          .message-box {
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #764ba2;
            white-space: pre-wrap;
            word-wrap: break-word;
          }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>📬 New Feedback Received</h1>
          <p>RECAP Project Hub</p>
        </div>
        <div class="content">
          <div class="field">
            <div class="label">From:</div>
            <div class="value">${name}</div>
          </div>
          <div class="field">
            <div class="label">Email:</div>
            <div class="value"><a href="mailto:${email}">${email}</a></div>
          </div>
          ${subject ? `<div class="field"><div class="label">Subject:</div><div class="value">${subject}</div></div>` : ''}
          <div class="field">
            <div class="label">Message:</div>
            <div class="message-box">${message}</div>
          </div>
        </div>
      </body>
      </html>
    `;
    
    // Send admin notification
    console.log(`📤 Sending email to ${recipientEmail}...`);
    
    try {
      await transporter.sendMail({
        from: `"RECAP Feedback" <${process.env.SMTP_USER}>`,
        to: recipientEmail,
        replyTo: email,
        subject: feedbackSubject,
        html: htmlContent
      });
      console.log('✓ Admin notification sent');
    } catch (emailError) {
      console.error('❌ Failed to send admin email:', emailError);
      throw new Error(`Email sending failed: ${emailError.message}`);
    }
    
    // Send user confirmation
    try {
      await transporter.sendMail({
        from: `"RECAP Team" <${process.env.SMTP_USER}>`,
        to: email,
        subject: 'Thank You for Your Feedback - RECAP',
        html: `
          <!DOCTYPE html>
          <html>
          <body style="font-family: sans-serif; padding: 20px;">
            <h2 style="color: #667eea;">✅ Thank You, ${name}!</h2>
            <p>We've received your feedback and will get back to you soon.</p>
            <p style="color: #6c757d;"><em>"${message.substring(0, 150)}${message.length > 150 ? '...' : ''}"</em></p>
            <p>— The RECAP Team</p>
          </body>
          </html>
        `
      });
      console.log('✓ User confirmation sent');
    } catch (confirmError) {
      console.warn('⚠️ Failed to send confirmation email:', confirmError.message);
    }
    
    // Store in Firestore
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
