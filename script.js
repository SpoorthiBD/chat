function sendMessage() {
    var userInput = document.getElementById("user-input").value;
    if (userInput !== "") {
        appendUserMessage(userInput);
        // Here you can add code to send the user message to your backend for processing
        // and receive the bot's response
        // For demonstration purposes, let's just simulate a bot response after a short delay
        var request = new Request("http://127.0.0.1:5000/chat")
        fetch(request, {
          method: "POST",
            body: JSON.stringify({
              "message": userInput
            }),
            headers: {
              "Content-type": "application/json; charset=UTF-8",
              "Access-Control-Allow-Origin": "*"
            }
          })
        .then((response) => response.json())
        .then((json) => {
          //console.dir(json)
         var botResponse =  json.response;
            appendBotMessage(botResponse);
        });
              
        
        setTimeout(function() {
           
        }, 500);
        document.getElementById("user-input").value = "";
    }
}

function appendUserMessage(message) {
    var chatBox = document.getElementById("chat-box");
    var userMessageElement = document.createElement("div");
    userMessageElement.classList.add("chat-message");
   userMessageElement.innerHTML = "<p class='user-message'>" + message + "</p>";
    chatBox.appendChild(userMessageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function appendBotMessage(message) {
    var chatBox = document.getElementById("chat-box");
    var botMessageElement = document.createElement("div");
    botMessageElement.classList.add("chat-message");
    botMessageElement.innerHTML = "<p class='bot-message'>" + message + "</p>";
    chatBox.appendChild(botMessageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  
  displaySelectedButton = (link) => {
    // window.open(link,'_blank')
    const id = "layout-view";
    const div = document.getElementById(id);
    div.innerHTML = '<iframe style="width:100%;height:100%;" frameborder="0" src="' + link + '" />';
  }