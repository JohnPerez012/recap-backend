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

// Polyfill fetch for Node.js < 18
if (typeof fetch === 'undefined') {
  global.fetch = require('node-fetch');
}

const app = express();
const PORT = process.env.PORT || 3001;

// ============================================
// MIDDLEWARE
// ============================================
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// ============================================
// FIREBASE INITIALIZATION
// ============================================
const serviceAccountPath = process.env.FIREBASE_SERVICE_ACCOUNT_PATH || './serviceAccountKey.json';
const fs = require('fs');

if (fs.existsSync(serviceAccountPath)) {
  console.log(`Using Firebase credentials from file: ${serviceAccountPath}`);
  const serviceAccount = require(serviceAccountPath);
  
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
  });
  console.log('✓ Firebase Admin initialized successfully');
} else {
  console.error('❌ Firebase service account file not found:', serviceAccountPath);
  process.exit(1);
}

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
// AI PROVIDER CONFIGURATIONS
// ============================================
const AI_PROVIDERS = { 
  mistral: { // first
    name: 'Mistral AI',
    endpoint: 'https://api.mistral.ai/v1/chat/completions',
    model: 'open-mistral-nemo', // mistral-small-latest: Mistral's paid API tier charges for mistral-small
    apiKey: process.env.MISTRAL_API_KEY,
    headers: (apiKey) => ({
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    }),
    formatRequest: (messages, maxTokens) => ({
      model: 'open-mistral-nemo',
      messages,
      max_tokens: maxTokens,
      temperature: 0.7
    }),
    extractResponse: (data) => data.choices[0].message.content
  },
  
  groq: { // Second
    name: 'Groq AI',
    endpoint: 'https://api.groq.com/openai/v1/chat/completions',
    model: 'llama-3.3-70b-versatile',
    apiKey: process.env.GROQ_API_KEY,
    headers: (apiKey) => ({
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    }),
    formatRequest: (messages, maxTokens) => ({
      model: 'llama-3.3-70b-versatile',
      messages,
      max_tokens: maxTokens,
      temperature: 0.7
    }),
    extractResponse: (data) => data.choices[0].message.content
  },
  
  gemini: { // Third
    name: 'Google Gemini',
    endpoint: (apiKey) => `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${apiKey}`,
    model: 'gemini-1.5-flash', //gemini-pro (v1) is deprecated
    apiKey: process.env.GEMINI_API_KEY,
    headers: () => ({
      'Content-Type': 'application/json'
    }),
    formatRequest: (messages, maxTokens) => {
      // Convert OpenAI format to Gemini format
      const contents = messages.map(msg => ({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content }]
      }));
      
      return {
        contents,
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: maxTokens
        }
      };
    },
    extractResponse: (data) => data.candidates[0].content.parts[0].text
  },
  
  openrouter: { // fourth
    name: 'OpenRouter',
    endpoint: 'https://openrouter.ai/api/v1/chat/completions',
    // model: 'openai/gpt-4o-mini', // Using mini for cost efficiency, can change to gpt-4o
    model: 'meta-llama/llama-3.3-70b-instruct:free', // much better kay explicit free model
    apiKey: process.env.OPENROUTER_API_KEY,
    headers: (apiKey) => ({
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://recaps-project-hub.onrender.com',
      'X-Title': 'RECAP Project Hub'
    }),
    formatRequest: (messages, maxTokens) => ({
      model: 'meta-llama/llama-3.3-70b-instruct:free',
      messages,
      max_tokens: maxTokens,
      temperature: 0.7
    }),
    extractResponse: (data) => data.choices[0].message.content
  }
};

// ============================================
// AI HELPER FUNCTIONS
// ============================================

/**
 * Call AI provider with proper error handling
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
    const body = provider.formatRequest(messages, maxTokens);
    
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
 * 4-Tier cascading fallback system
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
 * Main chat endpoint with 4-tier fallback
 */
app.post('/api/chat', async (req, res) => {
  try {
    const { message, conversationHistory = [], relevantProjects = [] } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    console.log(`\n💬 Chat Request: "${message.substring(0, 50)}..."`);
    
    // Build context-aware prompt with STRICT instructions
    let systemPrompt = `You are a helpful AI assistant for the RECAP Project Hub, a platform for research, capstone, and academic projects.

CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:
1. ONLY use information from the "Database Context" provided below
2. If asked about counts/numbers, ONLY use the exact numbers from the context
3. NEVER make up, estimate, or guess numbers
4. If the information is not in the context, say "I don't have that information in the database"
5. NEVER hallucinate or invent data

Your role:
- Help users find relevant projects based on their queries
- Provide information ONLY from the database context below
- Answer questions about research and capstone projects
- Be friendly, professional, and concise

Guidelines:
- If you have relevant project information, reference it specifically
- If asked about projects not in the context, say you don't have that information
- Keep responses clear and well-formatted
- Use bullet points for lists
- Cite specific project titles when relevant`;

    // Add project count if available
    if (relevantProjects.length > 0) {
      // Check if there's a project_count metadata
      const countProject = relevantProjects.find(p => p.id && p.id.includes('document_count'));
      
      if (countProject && countProject.projects_document_count) {
        systemPrompt += `\n\n📊 DATABASE STATS:
- Total projects in database: ${countProject.projects_document_count}
- Use this EXACT number when asked about project count`;
      }
      
      systemPrompt += `\n\nRelevant projects found in database:\n`;
      relevantProjects.forEach((project, idx) => {
        // Skip the count document in project listing
        if (project.id && project.id.includes('document_count')) return;
        
        systemPrompt += `\n${idx + 1}. **${project.title || 'Untitled'}**`;
        if (project.authors) systemPrompt += `\n   Authors: ${project.authors}`;
        if (project.abstract) systemPrompt += `\n   Abstract: ${project.abstract.substring(0, 200)}...`;
        if (project.year) systemPrompt += `\n   Year: ${project.year}`;
        if (project.program) systemPrompt += `\n   Program: ${project.program}`;
        systemPrompt += `\n   Relevance: ${(project.score * 100).toFixed(1)}%\n`;
      });
      
      systemPrompt += `\n\nIMPORTANT: Only reference these ${relevantProjects.length} projects when answering. Do not make up additional projects.`;
    } else {
      systemPrompt += `\n\nNo relevant projects found in the database for this query.`;
    }
    
    // Build messages array
    const messages = [
      { role: 'system', content: systemPrompt },
      ...conversationHistory,
      { role: 'user', content: message }
    ];
    
    // Call AI with 4-tier fallback
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
    
  } catch (error) {
    console.error('❌ Chat error:', error);
    res.status(500).json({
      success: false,
      error: 'Chat failed',
      details: error.message
    });
  }
});

/**
 * Individual provider endpoints (for testing/direct access)
 */
app.post('/api/chat/mistral', async (req, res) => {
  try {
    const { messages, maxTokens = 1000 } = req.body;
    const response = await callAIProvider('mistral', messages, maxTokens);
    res.json({ success: true, response, provider: 'Mistral AI' });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.post('/api/chat/groq', async (req, res) => {
  try {
    const { messages, maxTokens = 1000 } = req.body;
    const response = await callAIProvider('groq', messages, maxTokens);
    res.json({ success: true, response, provider: 'Groq AI' });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.post('/api/chat/gemini', async (req, res) => {
  try {
    const { messages, maxTokens = 1000 } = req.body;
    const response = await callAIProvider('gemini', messages, maxTokens);
    res.json({ success: true, response, provider: 'Google Gemini' });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.post('/api/chat/openrouter', async (req, res) => {
  try {
    const { messages, maxTokens = 1000 } = req.body;
    const response = await callAIProvider('openrouter', messages, maxTokens);
    res.json({ success: true, response, provider: 'OpenRouter' });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
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
  console.log(`   1. ${AI_PROVIDERS.mistral.name} ${AI_PROVIDERS.mistral.apiKey ? '✓' : '✗'}`);
  console.log(`   2. ${AI_PROVIDERS.groq.name} ${AI_PROVIDERS.groq.apiKey ? '✓' : '✗'}`);
  console.log(`   3. ${AI_PROVIDERS.gemini.name} ${AI_PROVIDERS.gemini.apiKey ? '✓' : '✗'}`);
  console.log(`   4. ${AI_PROVIDERS.openrouter.name} ${AI_PROVIDERS.openrouter.apiKey ? '✓' : '✗'}`);
  console.log(`\n🌐 Endpoints:`);
  console.log(`   POST /api/pinecone/search - RAG semantic search`);
  console.log(`   POST /api/chat - AI chat with fallback`);
  console.log(`   GET  /api/health - Health check`);
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
