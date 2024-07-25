css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
}
.chat-message .avatar {
  width: 10%;
}
.chat-message .avatar img {
  max-width: 150px;
  max-height: 150px;
  border-radius: 10%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: white;
  display: flex;	
  flex-direction: row;
  align-items: center;
}
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2023/02/05/20/01/ai-generated-7770474_960_720.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2013/07/12/14/07/student-147783_960_720.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
"""
