(function () {
  "use strict";

  /**
   * Shows loader animation.
   */
  const displayLoader = () => {
    document.querySelector('#loader').style.display = 'flex';
  };

  /**
   * Hides loader animation.
   */
  const hideLoader = () => {
    document.querySelector('#loader').style.display = 'none';
  };

  /**
   * Hide loader on window load.
   */
  window.addEventListener('load', (e) => {
    hideLoader();
  });

  /**
   * Sends ajax post requests.
   * @param {String} url Request url
   * @param {Array} payload Request payload
   * @returns {Object} Response
   */
  const ajaxPost = async function (url, payload) {
    displayLoader();
    return await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "X-CSRFToken": Cookies.get('csrftoken'),
      },
      body: JSON.stringify(payload)
    }).then(response => {
      hideLoader();
      if (response.ok) return response.json();
      throw new Error(response.status);
    }).catch(error => {
      if (error.message == 403) {
        window.location.href = '/';
      } else {
        console.error(error.message);
        return false;
      }
    });
  };

  /**
   * Textarea event listeners.
   */
  document.querySelector('#content').addEventListener(
    'keyup', (e) => {
      // hide results
      document.querySelector('#results').style.display = 'none';
      // update character counter
      let characterCount = e.target.value.length;
      let counter = document.querySelector('#counter');
      let current = document.querySelector('#current');
      let maximum = document.querySelector('#maximum');
      current.textContent = characterCount;
      if (characterCount >= 950) {
        maximum.style.color = '#8f0001';
        current.style.color = '#8f0001';
        counter.style.fontWeight = 'bold';
      } else {
        maximum.style.color = 'inherit';
        current.style.color = 'inherit';
        counter.style.fontWeight = 'normal';
      }
    }
  );

  let content = '';
  let contentChanged = false;

  document.querySelector('#content').addEventListener(
    'change', (e) => {
      if (e.target.value.trim() != content) {
        contentChanged = true;
        content = e.target.value.trim();
      }
    }
  );

  /**
   * Process button event listener.
   */
  document.querySelector('#process').addEventListener(
    'click', (e) => {
      e.preventDefault();
      let text = document.querySelector('#content').value.trim();
      if (!text.length || !contentChanged) {
        document.querySelector('#results').style.display = 'block';
        return;
      }
      ajaxPost(
        e.target.dataset.url,
        {'text': text}
      ).then(response => {
        if (response && response.status == 'success') {
          contentChanged = false;
          createResults(response.data)
        } else {
          console.error(response.status);
        }
      });
    }
  );
  document.querySelector('#process').addEventListener(
    'submit', (e) => {
      e.preventDefault();
    }
  );

  /**
   * Creates results.
   * @param {Object} sents JSON object of results
   */
  const createResults = (sents) => {
    let results = document.querySelector('#results');
    results.innerHTML = '';
    
    sents.forEach((sent, idx) => {
      let se = '<p class="sent">#' + (idx + 1).toString() +'&nbsp;-&nbsp;&nbsp;';
      se += sent.sent + '</p>' + '<table class="token">';
      se += '<tr><th><u>Token</u></th><th><u>Lemma</u></th>';
      se += '<th><u>POS tag</u></th></tr>';
      se += '<tr><td></td><td></td><td></td></tr>';
      sent.tokens.forEach((token) => {
        se += '<tr><td>' + token[0] + '</td><td>' + token[1]; 
        se += '</td><td>' + token[2] + '</td></tr>';
      });
      results.innerHTML += se;
    });

    results.style.display = 'block';
  };


})();