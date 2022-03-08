chrome.runtime.onMessage.addListener(gotMessage);
function gotMessage(message,sender,sendresponse)
{
	console.log(message.sels);
	for (let i = 0; i < message.sels.length; i++) {
    	$(message.sels[i]).css('background-color', 'yellow');
    }
}