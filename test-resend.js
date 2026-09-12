/**
 * Test Resend API Integration
 * Run: node test-resend.js
 */

require('dotenv').config();

const RESEND_API_KEY = process.env.RESEND_API_KEY;
const RESEND_FROM_EMAIL = process.env.RESEND_FROM_EMAIL || 'onboarding@resend.dev';
const FEEDBACK_RECIPIENT_EMAIL = process.env.FEEDBACK_RECIPIENT_EMAIL;

console.log('\n🔍 Testing Resend API Configuration...\n');
console.log('RESEND_API_KEY:', RESEND_API_KEY ? `${RESEND_API_KEY.slice(0, 10)}...` : 'NOT SET');
console.log('RESEND_FROM_EMAIL:', RESEND_FROM_EMAIL);
console.log('FEEDBACK_RECIPIENT_EMAIL:', FEEDBACK_RECIPIENT_EMAIL);
console.log('\n');

if (!RESEND_API_KEY) {
  console.error('❌ RESEND_API_KEY is not set in .env file');
  process.exit(1);
}

if (!FEEDBACK_RECIPIENT_EMAIL) {
  console.error('❌ FEEDBACK_RECIPIENT_EMAIL is not set in .env file');
  process.exit(1);
}

async function testResend() {
  console.log('📧 Sending test email via Resend API...\n');
  
  try {
    const response = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${RESEND_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        from: RESEND_FROM_EMAIL,
        to: FEEDBACK_RECIPIENT_EMAIL,
        subject: 'Test Email - RECAP Backend',
        html: `
          <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
              <h1 style="color: #08D488;">✅ Resend API Test Successful!</h1>
              <p>If you're reading this, your Resend API integration is working perfectly!</p>
              <p><strong>Configuration:</strong></p>
              <ul>
                <li>API Key: ${RESEND_API_KEY.slice(0, 15)}...</li>
                <li>From: ${RESEND_FROM_EMAIL}</li>
                <li>To: ${FEEDBACK_RECIPIENT_EMAIL}</li>
              </ul>
              <p>You're all set to receive feedback emails! 🎉</p>
            </body>
          </html>
        `
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      console.error('❌ Resend API Error:');
      console.error(JSON.stringify(errorData, null, 2));
      process.exit(1);
    }
    
    const data = await response.json();
    console.log('✅ Email sent successfully!');
    console.log('📬 Email ID:', data.id);
    console.log('📨 Check your inbox at:', FEEDBACK_RECIPIENT_EMAIL);
    console.log('\n✅ Resend API is working perfectly!\n');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  }
}

testResend();
