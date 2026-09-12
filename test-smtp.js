/**
 * Quick SMTP Test Script for Brevo (Sendinblue)
 * Run: node test-smtp.js
 * Get credentials from: https://app.brevo.com/settings/keys/smtp
 */

require('dotenv').config();
const nodemailer = require('nodemailer');

console.log('\n🔍 Testing SMTP Configuration...\n');
console.log('SMTP_HOST:', process.env.SMTP_HOST);
console.log('SMTP_PORT:', process.env.SMTP_PORT);
console.log('SMTP_SECURE:', process.env.SMTP_SECURE);
console.log('SMTP_USER:', process.env.SMTP_USER);
console.log('SMTP_PASS:', process.env.SMTP_PASS ? '***' + process.env.SMTP_PASS.slice(-4) : 'NOT SET');
console.log('\n');

const transporter = nodemailer.createTransport({
  host: process.env.SMTP_HOST,
  port: parseInt(process.env.SMTP_PORT),
  secure: process.env.SMTP_SECURE === 'true',
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS
  },
  debug: true, // Enable debug output
  logger: true // Log information
});

console.log('⏳ Verifying SMTP connection...\n');

transporter.verify((error, success) => {
  if (error) {
    console.error('❌ SMTP Verification FAILED:', error.message);
    console.error('\n📋 Error Details:', error);
    process.exit(1);
  } else {
    console.log('✅ SMTP Verification SUCCESSFUL!');
    console.log('✅ Connection to Zoho Mail is working properly.');
    
    // Optional: Send a test email
    console.log('\n📧 Sending test email...');
    
    transporter.sendMail({
      from: process.env.SMTP_USER,
      to: process.env.FEEDBACK_RECIPIENT_EMAIL || process.env.SMTP_USER,
      subject: 'SMTP Test - RECAP Backend',
      text: 'This is a test email from RECAP backend. If you receive this, SMTP is working correctly!',
      html: '<p>This is a test email from <strong>RECAP backend</strong>.</p><p>If you receive this, SMTP is working correctly! ✅</p>'
    }, (err, info) => {
      if (err) {
        console.error('❌ Failed to send test email:', err.message);
        process.exit(1);
      } else {
        console.log('✅ Test email sent successfully!');
        console.log('📬 Message ID:', info.messageId);
        console.log('📨 Check your inbox at:', process.env.FEEDBACK_RECIPIENT_EMAIL || process.env.SMTP_USER);
        process.exit(0);
      }
    });
  }
});
