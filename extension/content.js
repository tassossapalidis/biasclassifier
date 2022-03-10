// highlights appropriate text within article
chrome.runtime.onMessage.addListener(gotMessage);
function gotMessage(message,sender,sendresponse)
{
	console.log(message.sels);
	for (let i = 0; i < message.sels.length; i++) {
		let color = '';
		if (message.bias_type == 'conservative') {
			color = '#FF7F7F'
		}
		else {
			color = '#9EC2FF'
		}
    	$(message.sels[i]).css('background-color', color);
    }
}