function RevealPointer(){
  return {
    id: 'RevealPointer',
    init: function(deck) {
      try {
        var doc = document;
        if (!doc) return;

        var pointer = doc.createElement('div');
        pointer.className = 'reveal-pointer';
        pointer.setAttribute('aria-hidden','true');
        Object.assign(pointer.style, {
          position: 'fixed',
          width: '18px',
          height: '18px',
          borderRadius: '50%',
          pointerEvents: 'none',
          background: 'rgba(255,0,0,0.9)',
          boxShadow: '0 0 8px rgba(255,0,0,0.7)',
          zIndex: 99999,
          transform: 'translate(-50%, -50%)',
          display: 'none'
        });
        doc.body.appendChild(pointer);

        var btn = doc.createElement('button');
        btn.className = 'reveal-pointer-toggle';
        btn.type = 'button';
        btn.title = 'Toggle pointer (or press P)';
        btn.innerText = 'Pointer';
        Object.assign(btn.style, {
          position: 'fixed',
          bottom: '18px',
          right: '18px',
          zIndex: 99999,
          padding: '8px 10px',
          borderRadius: '6px',
          border: 'none',
          background: 'rgba(0,0,0,0.6)',
          color: '#fff',
          fontSize: '13px',
          cursor: 'pointer',
          backdropFilter: 'blur(4px)'
        });
        doc.body.appendChild(btn);

        var active = false;

        function updatePointer(e) {
          var x = e.clientX || (e.touches && e.touches[0] && e.touches[0].clientX) || 0;
          var y = e.clientY || (e.touches && e.touches[0] && e.touches[0].clientY) || 0;
          pointer.style.left = x + 'px';
          pointer.style.top = y + 'px';
        }

        function enablePointer() {
          if (active) return;
          active = true;
          btn.style.background = 'rgba(0,150,0,0.85)';
          pointer.style.display = '';
          document.addEventListener('mousemove', updatePointer);
          document.addEventListener('touchmove', updatePointer, {passive: true});
        }
        function disablePointer() {
          if (!active) return;
          active = false;
          btn.style.background = 'rgba(0,0,0,0.6)';
          pointer.style.display = 'none';
          document.removeEventListener('mousemove', updatePointer);
          document.removeEventListener('touchmove', updatePointer);
        }

        btn.addEventListener('click', function(e){ e.stopPropagation(); if(active) disablePointer(); else enablePointer(); });

        document.addEventListener('keydown', function(e){
          try {
            var k = e.key;
            if (typeof k === 'string') {
              var lk = k.toLowerCase();
              if (lk === 'p' || lk === 'z') {
                if (active) disablePointer(); else enablePointer();
              }
            }
          } catch(err) {}
        });

        // update pointer when pressing/tapping to give immediate feedback
        document.addEventListener('pointerdown', function(ev){
          if (active) updatePointer(ev);
        }, true);

        // keep pointer state across slide changes
        if (deck && deck.on) {
          deck.on('slidechanged', function(){ /* keep state */ });
        }
      } catch(err) {
        console.warn('RevealPointer plugin error', err);
      }
    }
  };
}
