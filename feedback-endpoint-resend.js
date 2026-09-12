/**
 * FEEDBACK ENDPOINT USING RESEND API
 * Replace the existing /api/feedback endpoint in server.js with this code
 */

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
    
    // Check if Resend is configured
    if (!RESEND_API_KEY) {
      console.error('❌ Resend API not configured');
      return res.status(503).json({
        success: false,
        error: 'Email service is not configured. Please contact the administrator.'
      });
    }
    
    const feedbackSubject = subject || 'New Feedback - RECAP Project Hub';
    
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
    
    const PRIMARY_COLOR = '#08D488';
    
    // HTML email template for admin
    const adminHtmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
          }
          .email-container {
            max-width: 600px;
            margin: 0 auto;
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
              <div class="field">
                <div class="label">Attached Images (${images.length})</div>
                ${images.map((url, idx) => `<div style="margin-top:10px;"><a href="${url}" target="_blank">${url}</a></div>`).join('')}
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
    
    // HTML email template for user confirmation
    const userHtmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
          }
          .email-container {
            max-width: 600px;
            margin: 0 auto;
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
            margin-bottom: 20px;
          }
          .message-preview {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
            border-left: 4px solid ${PRIMARY_COLOR};
            text-align: left;
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
            <h1>✅ Feedback Received</h1>
          </div>
          <div class="content">
            <div class="greeting">
              <p>Thank You, <strong>${name}</strong>! 👋</p>
            </div>
            <p>We've received your feedback and will review it shortly.</p>
            <div class="message-preview">
              <p><strong>Your message:</strong></p>
              <p>${message}</p>
            </div>
          </div>
          <div class="footer">
            <p>© 2026 RECAP · CTU Daanbantayan Campus · Group 2 BSIT 3A</p>
          </div>
        </div>
      </body>
      </html>
    `;
    
    // Send emails via Resend API
    console.log('✓ Sending emails via Resend API...');
    
    // Send admin notification
    const adminEmailResponse = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${RESEND_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        from: RESEND_FROM_EMAIL,
        to: FEEDBACK_RECIPIENT_EMAIL,
        reply_to: email,
        subject: feedbackSubject,
        html: adminHtmlContent
      })
    });
    
    if (!adminEmailResponse.ok) {
      const errorData = await adminEmailResponse.json();
      throw new Error(`Admin email failed: ${errorData.message || adminEmailResponse.statusText}`);
    }
    
    console.log('✓ Feedback email sent successfully to', FEEDBACK_RECIPIENT_EMAIL);
    
    // Send user confirmation
    const userEmailResponse = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${RESEND_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        from: RESEND_FROM_EMAIL,
        to: email,
        subject: 'Thank You for Your Feedback - RECAP',
        html: userHtmlContent
      })
    });
    
    if (!userEmailResponse.ok) {
      const errorData = await userEmailResponse.json();
      console.warn('⚠️ User confirmation email failed:', errorData.message);
      // Don't throw - admin email succeeded
    } else {
      console.log('✓ Confirmation email sent to', email);
    }
    
    // Store feedback in Firestore
    try {
      await db.collection('feedback').add({
        name,
        email,
        feeling: feeling || null,
        subject: subject || null,
        message,
        images: images || [],
        timestamp: admin.firestore.FieldValue.serverTimestamp(),
        status: 'new'
      });
      console.log('✓ Feedback stored in Firestore');
    } catch (firestoreError) {
      console.error('⚠️ Failed to store feedback in Firestore:', firestoreError);
      // Don't throw - emails already sent
    }
    
    res.json({
      success: true,
      message: 'Feedback submitted successfully! Check your email for confirmation.'
    });
    
  } catch (error) {
    console.error('❌ Feedback submission error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to send feedback. Please try again later.'
    });
  }
});
