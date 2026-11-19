/* ══════════════════════════════════════════════════════════════════
   Figure 13 — The Neuroconnectionist Trifecta & opportunistic model
   experiments (Unified single-panel left-to-right pipeline):
   [1. Sensory Diet] => [2. Architecture] => [3. Objective / Latent Hypersphere]
   with a control-varying opportunistic model-zoo annotation card
   underneath the hypersphere.
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('trifecta_explorer', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 13,
    title: 'Types of deep learning models: sensory diet \u21d2 architecture \u21d2 objective',
    hint: 'Choose which Trifecta factor to vary &middot; click any opportunistic experiment card beneath the sphere or use the toggles &middot; press play or drag the 3D hypersphere'
  });

  var THUMB_URIS = {
    dogs: [
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAcFBQYFBAcGBgYIBwcICxILCwoKCxYPEA0SGhYbGhkWGRgcICgiHB4mHhgZIzAkJiorLS4tGyIyNTEsNSgsLSz/2wBDAQcICAsJCxULCxUsHRkdLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCz/wAARCAAwADADASIAAhEBAxEB/8QAGgAAAgMBAQAAAAAAAAAAAAAABAUDBgcIAP/EADEQAAIBAwIDBgUDBQAAAAAAAAECAwQFEQAhBhIxBxMiQVFhFHGBobEjMpEVFkLR4f/EABgBAQEBAQEAAAAAAAAAAAAAAAQDAgUG/8QAIREAAgICAgIDAQAAAAAAAAAAAQIAEQMSBCFBoTGBsfD/2gAMAwEAAhEDEQA/AKZy1MsyLCfEp5woGc+f40Z+qkOJCGmUArGrDLEdQfTz0s72pQgQVBiEnhkRW3Yg+fsD030yp1V5UjkcM4HKzsBsMb499eayCqmDCaKGGpqkTvJJsMC2Bnrk4B8zosUsVMB8THJA5AHKykHPn00z4SpJqioppaNMU6vKskr9VCD8knVsktMT2+oSTmklNScSPv4OTOB9To7EK+jGuo/Fxdl2MzoRQU5buGRFOWyBuSSN86gqI4VJ5oi0vIVVum56npvpXfeIZaG/V8D0wjETKCAOUeo26A6Lp6lrqkaUrPJ07s4Jc825A9cHb11dsDpTHzCupU1ArUkPeCNUXPLkMq5Ow6gf7204t1CjVJJbDqrY9yFJHXruPxrySR2uCGJX56iRBmUOI+8yei+xzjPn5aDeRRDIedhLzB1yTzKfLfWnYk2JPpSJcuDLZNfeEf6dS3GpoqoSswkhblL+INg+xyNvbVs48tt4snCtOLeEa4SAp4jhQx8x9BpH2avVQ193rLfDzmloGqI6dhjMmPCv5/gagsPH1746tN5huEAlp6JEaGcRcjd5zbrtsTjfVW4YbfK/fdj0T+TpjkFSqj+8TMeMGrFtbT3V4XuFV3aSGIYBKliPscaS2Hi+58LUqT0Miczu4AcHYYGSD5ak41rmqeIKimzlIJCB88D8YA1Wq4/qxwhspCgUfPqfuTrpYcQOMK4+e4bMwZjU0FpHVi8SNyI+ctkeIe/8baaRQd5Sx1R8ck+CqjJYMP8Avl89OeFuzK88URmsUQUdtk/bUVBJaQA/4oOo98ga0CP+3uy+2MaZvjbtIrL8TKBzNtkhQNlG3T76AcViz0IMDyYNwBwVdKS8RXeqr5LXTzQdy1K+A8yYwRg/t6AgnfbppxU2i322re3UFero7c00QxzhSdyAPXffWUXXtCudTVA8pjkOAzOSeXJPT0OhOHrtVW3jCnvLFqhwcyjOQ653B+2NUXIFXUym19yiX2irk4kuNXWW+WBPiJGCOMYyxKg/TGqxM/NM7N1JzrrrtE4bpOJuB5Lza4FkSWATAKPENs/b01yZU0EwrpI5EaIq2GDDBB+WnYMjMDuKPr6lX1oamf/Z',
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAcFBQYFBAcGBgYIBwcICxILCwoKCxYPEA0SGhYbGhkWGRgcICgiHB4mHhgZIzAkJiorLS4tGyIyNTEsNSgsLSz/2wBDAQcICAsJCxULCxUsHRkdLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCz/wAARCAAwADADASIAAhEBAxEB/8QAGwAAAgMAAwAAAAAAAAAAAAAABQcDBAYAAQL/xAAvEAACAQIEBQIGAQUAAAAAAAABAgMEEQAFEiEGEzFBUQdhFCJxgZGhFjIzkrHw/8QAGQEBAQADAQAAAAAAAAAAAAAAAwQAAgUB/8QAHxEAAgIDAQADAQAAAAAAAAAAAQIAAxESMSEEE0FR/9oADAMBAAIRAxEAPwBMw0tQNaQyxCGc/M7Dpb/v3iM08tMGvGZEcWRugvte30v+8HcoyvMs20qiimQgNrYWQqT298CaxaqKpmpKhQssL6UjJstu5GBDZMBe9gCeI85iQyi9gT5GCuRZTWZxVw0tO0RaVtCmWQIpYmwGo9z4x6p4pah41dVIeQWt8wG/cYYnC3DyZdFVZhU0TRShHSlBYxDm2Nje+wvt9d+2PXs1Eesfa+hEz+e8AZpwrlkhzeRUkCmSOOBjIjHUFtqG1xcE+1sZ6jyNIW+JrIxPCyagig7E+cbn+QVEfp4EzfLFimqKvkxSysxciOKxO53tdVv3tgJS5vFByoaQgTq2osbb97b9Re3XxjMtwwr8BtU8EmqM1qWiblsym2qw264D18hrIfi5SedcIoNz9vphjT5fK6GJaCInppCgbYFycHTSl6mukeGBDzLoFQ2Ft7dBb87e+JanA7DrALedgXh3geepenqJ546SmdQTM8hUx3YCwsRckHp9MPWj4cNdBDQPd40AR4zZjZdgbn2thFiPMM6zY1E3MFKHIi1AhD777E+Th/8ACVYwmotU6vIihZLN2A6j7f6xtYpOuxnWoGinz2KX12pJKTiXLcpUiCmiy8SxIoADuZG1E/4j8DGF4UyamzQ1BqpmhZQBEYyCCx2t5w7PUXgubjHi96+SfQsES08dz2BJP7Y/rA7L/Sn4ZtRzBFRTcctbEEdN8bPZ1QZy2bLS6pLf3NRO1ugF8VAlPnNc1AWWZkYa4w3fqQfFhvi3NlBm182KMGQAMqMykjz8rC2OZZk5yjOHr6eCPmGJoViIsqAkb7bnp5xHqP7H+MtVNm59xDddDluYZIuUlwppauMQgKToIW7KPYr1+2C9LFRU9d8JTaFkWLWUTYLYG1/zgGzTIjvTI8FaTr56lSASbkBSCADjrI6SbLs4FW85mUseYtralP8AVbx7YcOqqFEpe7azI5C9dXoa+pHNUHmbLbp3xGJNcgMkxCk3HWzYhr6VKiqjlTmCWw5hsNLhRYG3Y2Avvv7YgeCSchQSh7A+cE2MyfC/s//Z',
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAcFBQYFBAcGBgYIBwcICxILCwoKCxYPEA0SGhYbGhkWGRgcICgiHB4mHhgZIzAkJiorLS4tGyIyNTEsNSgsLSz/2wBDAQcICAsJCxULCxUsHRkdLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCz/wAARCAAwADADASIAAhEBAxEB/8QAGgAAAgMBAQAAAAAAAAAAAAAABQYCBAcDAf/EADMQAAIBAgQFAQYEBwAAAAAAAAECAwQRAAUSIQYTMUFRcQcUImGBoRUyQtEXUnKCkbHx/8QAGQEBAAMBAQAAAAAAAAAAAAAAAgEEBQMG/8QAHxEAAgICAwADAAAAAAAAAAAAAQIAEQMhBBITMUFR/9oADAMBAAIRAxEAPwB0QrG2lVUeAH6/8xLS+hiqEHbcv/vCN/EPhyON5Y1r6pNWkSRwW7eCfGGDI+I8q4mozLl88wZVDssq6WG9gTba3Xp4x5lsTqLYTWDqdCEKueGmieWonhgTYapJgAcQoK3L6+Z0pMwp5mSzEQTh9PqAdsZn7TMseHiOjrhVtLDPIYlQncMFG47W3++BHCLVFP7RcrFNYNPrhkC7Bksbg/MHf6Y7rxlbH3BgOQ3VTcllCLcaRp3udyfH2Bx5HViQlQwQ3sN7YRuN+N5eE6yCjoqOOSaRAxnkuY4wTbp+o7H5DCxTe0HNaOsiNYRW00p+OPkrG4uPzIw6+bHtgJx8jL2ERyKDRi/HQqkFbJS1Rkp6ecxxtsrPfuB2Ft8Csh40zPhbMHWlWKaGSQrLCw2kAbpfqO9j88X6+l90kbKzCzO83Ji0kq17GwBv31emAOb8N5jlNaU93ZmhI1hPjCNuwF+5spJ8Y10Aaw+7lR9UVmj8YcS0md1eXLRqFheL3izmxDN5PUadIHrhazTPKaPiqnr8kQ0Zo4UcIfyvIL6hcHcEnc98UqfJ6qSmkhq3COkbs5BGpNJ1WI6gdcUqnL6lcshEKiZmk5KCIXZv3wMeNF0ImyuVqpoHEPE1FxVwbTusOitknSORB+lVOplv/KTY/UeMB4638G4mpJs4pEipjzCyx/EFXlso2G/f7YBNlE1Ll8LlSEjVBUOjagC1ypFvHQ/XFaqbMK+oeaV5J1ik5Ksxvc3sAP8AF8BcK/CnW4vQhdjeppud5DNW51QVNPVFZopY5VuoFreftg02Ux8jMpKlkmeWfWq2KgAAqL23NwbH5DHeq5SzVEqBTFHGEVIxZiSDt8unX0xb5iowdpC0gjS8ajZ7m3Xzv9bYzPU0BL3QXAVDwzlq1FTWNRrLKzmMvrttckgD+7v++IZdw1kuWUULzISkdQZ4oSbJdoyoFyb2+En1weNO8RjiRNPxkshuzgkd73v64pyoKmnMCFpzSyKpsdQstrgHubEi3fEjMx1cgqPyKYos0zTNa/kSJT0R5cSALdNQsBcWPTci3nHaTh+QVa5Q8iy0cSmokmVLEyrY6v6bN37DDIKR4Pd4lgPLEl9JGkAk283vY74vingpqoo5jKEBJXU9eg8dML1P1J6if//Z'
    ],
    cats: [
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAcFBQYFBAcGBgYIBwcICxILCwoKCxYPEA0SGhYbGhkWGRgcICgiHB4mHhgZIzAkJiorLS4tGyIyNTEsNSgsLSz/2wBDAQcICAsJCxULCxUsHRkdLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCz/wAARCAAwADADASIAAhEBAxEB/8QAGgAAAwEBAQEAAAAAAAAAAAAABAYHBQMCCP/EAC0QAAEDAwMDAgYCAwAAAAAAAAECAwQABREGEiETMUEUUQciMmFxkRUjQoGx/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKDq1gO2F8Ac7ajGoZAj2iW20tSp6khLTbffkjx98gfurrfEBVqdB7bTUYtN0n2tcmbd2Iv8erqKdByXmm9wCVgdlAnAwOeQaDparRcGra0mRb0JUCkOJCwoJxg5BHj7dxg1TpS7ZfbI7IhPNvJxtWkcKQr2UO4NTK46qlrgBNjSmU+8ha2Ax/YOmFBPU47cnsecgiiXOto+THZZdS5dZMfDiXgpe9Lac7Tg8DlSs5J3LA7A0DHZrBuYWggnB4zXqRp7Cj8lcbX8R7fbIHqbla5re88hlKXEj8nIp4ssyBqiyM3a3hZjP529RO1QIOCP2KAm5o3wHB7pNJCbBE1XpqRbVrVFkAhKJCRnapJBSSPIyBkU8zXUenLZUN6xwPJqcs3Wda75KgRWjyrLjhGS0T2wOx96A74e6Uc0jZJJuiWjLU+4pbiTu3IB+XBx5wVflVLcd6JdtbXK9NurdYUSy0pxe7elP1lCf8UhXy58kU2XS1zNbfD9+EZMizPOJI3duqU54PnpqOCcckVNNHRZejZ3p53QkpuEjpLjMciOdp2KBxg7iNpA+xNBoaVszutdQTbPHSYsCM+UyX8ZAGcgNn3Psa+golvi2yCxBhspZjR0BtttPZKR2pK0PBRaQzEQkIOCtQAxlROT/wBx/qn1z6qCVT9Vwol0kOTH9iGsJGO/Azx7nmh7ZKRcXnri80A4+N3AxwBwKmN6lSrrqBTTawIbq0uKJ8kAcfuni1TUtoaZ3cIwSfxQb9u9c/afS73YpPAUPqHnvWhp/RMCLdUXCUESHWnOqwnYEhpeOSMfk/uh412QMciteLdEZBKsCgJQj0mr0t4wl1JWj7jzTY54pOuUluQuJLQ6EuxnNw+47Efo01JkpW2k57ig/9k=',
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAcFBQYFBAcGBgYIBwcICxILCwoKCxYPEA0SGhYbGhkWGRgcICgiHB4mHhgZIzAkJiorLS4tGyIyNTEsNSgsLSz/2wBDAQcICAsJCxULCxUsHRkdLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCz/wAARCAAwADADASIAAhEBAxEB/8QAGgAAAgMBAQAAAAAAAAAAAAAABQYCBwgEA//EACwQAAIBAgQFBAICAwAAAAAAAAECAwQRAAUSIQYTIjFBFFFxkWGBBxVCoeH/xAAWAQEBAQAAAAAAAAAAAAAAAAAFBAP/xAAgEQACAgIBBQEAAAAAAAAAAAABAwACBBESEyEiMUFC/9oADAMBAAIRAxEAPwDPdNDVV8609HTyTzMNkiUsx+AMeseX11LUOJ6SohMY6w8bLpv2vfEMozWpyTNqfMaNtM9O+pb9j7g/gi4xpfhDizKuK+FzU8qPXYJPG4DFWH+J9x5GJ3N6fsdpupXU9e5nOgp5HqhHHGzyOQFAFyfjDanAPEQCVP8AXSWdx0m2r6xeMmVZLHVQ1MGX0yVEC6Y2jjVdP1jryWVHhqpKoWmCsYgTsoHi3v5vgpmabW40ETVi8a87Sv8AIOFKqdKf1DemEeoHV5NyQMHpuB8tqapFr5JZAlm0qQouPOHqDKxHSXaO8dQLOTv1EDf7ws1TtThWka5iJ6j5W9j+wf8AWImA11b7LVsN91B7TJuLA/iF53z6upo53SOSm1Mi76yGFtv3hAKsPBxaX8P5I8S12eT3RNPIiBB6jcFjt+hh7NsAi24JhVNn11LfSLTTos4VVkFg0iFwGv5C7jEquo9BJRSuQqyg3kj2jB2uDfcEG+x7gjEIYY6ioQ9Q5R6mYiMDY7Asbn9Y682qUiq6eikgapkGl3p0GoqD2ue1/wAYAqBxjlj56jXTzSPlJXWJEY3Vgdj7YXc/olaCoI6SkpW1vcC4+jbB3JqDMY6GgpqpIlhkZpTGN3QX1BSfn67YF8QViLV5jTLp5iTc1ixFx0A+Pzi1y9IBMiSeLyKntKuyzhn1rqlNSqx99AAGGmXhwZWtOhLcxN2IJtYHtpGw3Pzjv4ZkC0sfLm30mwFtr+/ucF6uJ51ISp5LXuSUDX+8bZdGOGvkxxGUSdiBZJI4qqJ3WCRG2Ebxhy5tfuRtgxQ5fPPWGreQapIho17bXtf8C+Bs2WVj1kTq9O4jlDKDqHztgvNDmstWjN6UKqaQodrW8X2wdRDP0Ig11NeJEYRK1PyeU3MkQHSCfNrb4rapyuqn4kzBJZAJaia8hXcCNQAd/wAnYfBw+ZbSzmojkqnh1pfSFvb/ALgQconhkrXjzGA1E8rOzNA2wPZRv2GLG0ZdYFRI0Xoq5NjP/9k=',
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA8Q1JFQVRPUjogZ2QtanBlZyB2MS4wICh1c2luZyBJSkcgSlBFRyB2NjIpLCBxdWFsaXR5ID0gOTAKAP/bAEMABwUFBgUEBwYGBggHBwgLEgsLCgoLFg8QDRIaFhsaGRYZGBwgKCIcHiYeGBkjMCQmKistLi0bIjI1MSw1KCwtLP/bAEMBBwgICwkLFQsLFSwdGR0sLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLP/AABEIADAAMAMBIgACEQEDEQH/xAAaAAADAQEBAQAAAAAAAAAAAAAFBgcDBAEC/8QAMxAAAgECBAQDBgUFAAAAAAAAAQIDBBEABRIhBhMxQVFhcQcUIkKBkRUXYsHwIyQzsdH/xAAYAQADAQEAAAAAAAAAAAAAAAABAgMABP/EABsRAAMBAAMBAAAAAAAAAAAAAAABAhEhMUED/9oADAMBAAIRAxEAPwAHmJQtTli6oFY/Dsb3269v2xrQ5YsjywtLFT0yskkiSsBIpDXAHiNtiP2wLzOpq3FGHpuQrSaNRYHY7WsPXGWeZi9SKGspeWje78rlg6mAvszHxuL+WIVCqlpRZg7+7ZXS5slRJM0kqggMdgEuW38d2vjI57JX8R0+uFYqKphkWLSLNYC4Y+p7Yn1dmlc8Bi1f5A0TnqbEWODlDmgkrsngSVqmKmBUhowrJ02Ldwd8C/jL6RSbaWA/ikFXgnUXkiH3Hf8AnlhkzijoG4NijSnVfe4Y21AmxARnO3npH3wHzyHm7RnW6fFYbnbHyEnWWjy+Wp/t3pA1MZFuLuQCl/03cb+QxK1rQs+jA8Mc9XRrKARCxka/Q2BsPv8A6wBzaKlSzNA8UbsWMlho02PSx23wRr6s00MTAbGddZJ6qRY/TB7JeHqjibPqSmiZaRoX1Sum4EXcDsR/3HUKmIdZktZQUlDXTQuYswiZ4SFLFWDFdJHjcWwVy9xC0NS0UMayfIragCOx7Xxba/h7Kc8hHC4oaqBcspysdYDpClrgLf5tQuxt0+uIzxLE/Dr1FBNyylMNHM0E7g7WI79vTBN1wxMmzqKKsqaSdCyxSuI3VrMNzbAufinMXkhKzC0AKx/pF74F1ZtK0nM1FySW9cc6oxjLjoMFpMTSr5+inL4ZNN9TaCPHuMVf2bZP+K8CQ1yStTVsc8kaTIeqgjY+V74lOagtlsRuNCkffpi1exdo/wAuxHt/SqpQfqQcJ6BvkR/aBV5vwaq1sU8olqGs8vOaxt2sdvTbElzfO5c7dJmLQkA69DaSvmT81/PFD9u2dfiedw0cEg5NHcWXf4z3+2JPLakpCzD43NzfqTjSOzgzALJUqikeJJ7YyYIkDxcyzKbb98eMTJVhrdBc2xjUkiofrcbk+f8ADhxT/9k='
    ]
  };

  var thumbImgs = { dogs: [], cats: [] };
  ['dogs', 'cats'].forEach(function (kind) {
    THUMB_URIS[kind].forEach(function (uri) {
      var im = new Image();
      im.onload = function () { if (ready) drawAll(); };
      im.src = uri;
      thumbImgs[kind].push(im);
    });
  });

  var VALID_OBJ = ['supervised', 'autoencode', 'contrastive', 'clip'];
  var varyAxis = cfg.vary || 'task';
  var taskMode = VALID_OBJ.indexOf(cfg.objective) >= 0 ? cfg.objective : 'contrastive';
  var archMode = 'cnn';   // 'cnn' | 'vit'
  var dietMode = 'both';  // 'dogs' | 'cats' | 'both'
  if (varyAxis === 'arch') { taskMode = 'supervised'; dietMode = 'both'; }
  if (varyAxis === 'input') { taskMode = 'contrastive'; archMode = 'cnn'; }

  var prog = 1.0; // 0..1 training progression
  var yaw = -0.32, pitch = 0.24, dragOrigin = null;
  var sphereBounds = { cx: 0, cy: 0, R: 80 };
  var ready = false;

  /* 24 instances: 12 Dogs (family 0, blue) + 12 Cats (family 1, orange) */
  var N = 24, initPts = [], initTwins = [], fam = [];
  (function buildInit() {
    var rng = M.rng(cfg.seed || 53);
    for (var i = 0; i < N; i++) {
      fam.push(i % 2);
      var v = M.normVec([rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)]);
      var tw = M.normVec([v[0] + rng.normal(0, 0.50), v[1] + rng.normal(0, 0.50), v[2] + rng.normal(0, 0.50)]);
      initPts.push(v);
      initTwins.push(tw);
    }
  })();

  var SUP_CENTERS = [
    M.normVec([0.76, 0.34, 0.55]),   // Dog prototype
    M.normVec([-0.76, -0.28, 0.58])  // Cat prototype
  ];

  function targetPositions(mode) {
    var rng = M.rng(909 + (archMode === 'vit' ? 17 : 0));
    var pts = [], twins = [];
    var spreadSup = archMode === 'vit' ? 0.13 : 0.19;
    for (var i = 0; i < N; i++) {
      var f = fam[i], u = i / N;
      if (mode === 'supervised') {
        var c = SUP_CENTERS[f];
        var p1 = M.normVec([c[0] + rng.normal(0, spreadSup), c[1] + rng.normal(0, spreadSup), c[2] + rng.normal(0, spreadSup)]);
        pts.push(p1); twins.push(p1);
      } else if (mode === 'autoencode') {
        var lon = (u - 0.5) * Math.PI * 1.75;
        var lat = (f === 0 ? 0.34 : -0.34) + 0.14 * Math.sin(u * 8);
        var p2 = M.normVec([Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)]);
        pts.push(p2); twins.push(p2);
      } else if (mode === 'contrastive') {
        var phi = Math.acos(1 - 2 * (i + 0.5) / N);
        var theta = Math.PI * (1 + Math.sqrt(5)) * i;
        var p4 = M.normVec([Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta)]);
        var tw4 = M.normVec([p4[0] + rng.normal(0, 0.05), p4[1] + rng.normal(0, 0.05), p4[2] + rng.normal(0, 0.05)]);
        pts.push(p4); twins.push(tw4);
      } else {
        /* CLIP: paired image + text vectors converge around semantic caption regions */
        var c5 = SUP_CENTERS[f];
        var p5 = M.normVec([c5[0] + rng.normal(0, 0.22), c5[1] + rng.normal(0, 0.22), c5[2] + rng.normal(0, 0.22)]);
        var cap = M.normVec([p5[0] + rng.normal(0, 0.06), p5[1] + rng.normal(0, 0.06), p5[2] + rng.normal(0, 0.06)]);
        pts.push(p5); twins.push(cap);
      }
    }
    return { pts: pts, twins: twins };
  }

  /* ─── DOM: Single Unified Left-to-Right Stage + Control-Varying Annotation Underneath ─── */
  M.mini(root, 'neuroconnectionist pipeline: sensory diet \u21d2 architecture \u21d2 objective / latent hypersphere');
  var stage = M.stage(root, {
    height: function (w) { return w < 620 ? 500 : M.clamp(w * 0.42, 278, 338); },
    ref: 760,
    label: 'Single-panel left-to-right diagram of Sensory Diet flowing into Computational Architecture flowing into the 3D Latent Hypersphere'
  });

  M.legend(root, [
    { color: C.blue, label: 'dog exemplars' },
    { color: C.orange, label: 'cat exemplars' },
    { color: C.green, label: 'supervised label / paired text anchor' }
  ]);

  /* Control-varying model-zoo annotation card directly underneath the hypersphere */
  var zooCard = root.appendChild(M.h('div', { class: 'trifecta-zoo-card', 'aria-live': 'polite' }));

  var tr = M.transport(root, {
    n: 61, fps: 30, aria: 'Training progression from random initialization to learned geometry',
    fmt: function (k) { return 'training ' + Math.round((k / 60) * 100) + '%'; },
    onFrame: function (k) { prog = M.ease(k / 60); drawAll(); }
  });
  tr.show(60);

  var row1 = M.row(root);
  var varyT = M.toggle(row1, {
    label: 'Vary 1 factor', value: varyAxis,
    options: [
      ['task', 'Task Objective'],
      ['arch', 'Architecture'],
      ['input', 'Sensory Diet']
    ],
    onChange: function (v) {
      varyAxis = v;
      if (v === 'arch') { taskMode = 'supervised'; dietMode = 'both'; }
      else if (v === 'input') { taskMode = 'contrastive'; archMode = 'cnn'; }
      else { dietMode = 'both'; }
      rebuildSubControl();
      renderZooCard();
      tr.show(60);
      drawAll();
    }
  });
  M.spacer(row1);
  M.resetButton(row1, function () {
    tr.pause();
    varyAxis = 'task'; taskMode = 'contrastive'; archMode = 'cnn'; dietMode = 'both';
    yaw = -0.32; pitch = 0.24;
    varyT.set('task', true);
    rebuildSubControl();
    renderZooCard();
    tr.show(60); drawAll();
  });

  var row2 = M.row(root);
  var read = M.readout(root);
  var subToggle = null;

  function rebuildSubControl() {
    row2.innerHTML = '';
    if (varyAxis === 'task') {
      subToggle = M.toggle(row2, {
        label: 'Objective (varied)', value: taskMode,
        options: [
          ['supervised', 'Supervised'],
          ['autoencode', 'Autoencoder'],
          ['contrastive', 'Contrastive'],
          ['clip', 'CLIP (Vision + Text)']
        ],
        onChange: function (v) { taskMode = v; renderZooCard(); tr.show(60); drawAll(); }
      });
      M.label(row2, '&middot; Held: Dogs+Cats &amp; CNN/ViT');
    } else if (varyAxis === 'arch') {
      subToggle = M.toggle(row2, {
        label: 'Architecture (varied)', value: archMode,
        options: [
          ['cnn', 'CNN (2D Conv)'],
          ['vit', 'Transformer (Self-Attn)']
        ],
        onChange: function (v) { archMode = v; renderZooCard(); tr.show(60); drawAll(); }
      });
      M.label(row2, '&middot; Held: Dogs+Cats &amp; Supervised');
    } else {
      subToggle = M.toggle(row2, {
        label: 'Sensory Diet (varied)', value: dietMode,
        options: [
          ['dogs', 'Dogs only'],
          ['cats', 'Cats only'],
          ['both', 'Dogs + Cats']
        ],
        onChange: function (v) { dietMode = v; renderZooCard(); tr.show(60); drawAll(); }
      });
      M.label(row2, '&middot; Held: CNN &amp; Contrastive');
    }
  }

  function renderZooCard() {
    var spec;
    if (varyAxis === 'task') {
      spec = {
        head: 'Opportunistic Model-Zoo Control — Vary Objective (Hold Dogs+Cats Diet & Architecture Constant)',
        cur: taskMode,
        onSelect: function (k) { taskMode = k; if (subToggle) subToggle.set(k, true); renderZooCard(); tr.show(60); drawAll(); },
        arms: [
          { key: 'supervised', col: C.blue, title: 'Supervised (e.g. Torchvision ResNet-18)', desc: 'Explicit category labels pull Dog & Cat poles apart.' },
          { key: 'autoencode', col: C.purple, title: 'Autoencoding (e.g. MAE / Conv-AE)', desc: 'Reconstruction organizes smooth pose & appearance manifolds.' },
          { key: 'contrastive', col: C.green, title: 'Self-Supervised Contrastive (e.g. SimCLR)', desc: 'Instance augmentations pull views together without labels.' },
          { key: 'clip', col: C.orange, title: 'Vision-Language Contrastive (e.g. OpenCLIP)', desc: 'Joint image\u2013caption cosine alignment on the unit sphere.' }
        ]
      };
    } else if (varyAxis === 'arch') {
      spec = {
        head: 'Opportunistic Model-Zoo Control — Vary Architecture (Hold Dogs+Cats Diet & Supervised Objective Constant)',
        cur: archMode,
        onSelect: function (k) { archMode = k; if (subToggle) subToggle.set(k, true); renderZooCard(); tr.show(60); drawAll(); },
        arms: [
          { key: 'cnn', col: C.blue, title: '2D Convolutional Backbone (ResNet-18 / ConvNeXt, timm)', desc: 'Local sliding 2D filters detect edges, textures, and object parts across space.' },
          { key: 'vit', col: C.green, title: 'Self-Attention Backbone (ViT-B/16, timm)', desc: 'Global multi-head attention routes context across all image patches.' }
        ]
      };
    } else {
      spec = {
        head: 'Opportunistic Model-Zoo Control — Vary Sensory Diet (Hold CNN Architecture & Contrastive Objective Constant)',
        cur: dietMode,
        onSelect: function (k) { dietMode = k; if (subToggle) subToggle.set(k, true); renderZooCard(); tr.show(60); drawAll(); },
        arms: [
          { key: 'dogs', col: C.blue, title: 'Narrow Dogs-Only Diet (Open-IPCL)', desc: 'Allocates representational capacity to fine-grained within-dog variation.' },
          { key: 'cats', col: C.orange, title: 'Narrow Cats-Only Diet (Open-IPCL)', desc: 'Allocates representational capacity to fine-grained within-cat variation.' },
          { key: 'both', col: C.green, title: 'Diverse Dogs + Cats Diet (Konkle & Alvarez, 2022)', desc: 'Carves broader between-category axes alongside instance structure.' }
        ]
      };
    }

    zooCard.innerHTML = '';
    zooCard.appendChild(M.h('div', { class: 'trifecta-zoo-head', text: spec.head }));
    var grid = zooCard.appendChild(M.h('div', { class: 'trifecta-zoo-grid' }));
    spec.arms.forEach(function (arm) {
      var isAct = arm.key === spec.cur;
      var btn = M.h('button', {
        type: 'button',
        class: 'trifecta-zoo-arm' + (isAct ? ' active' : ''),
        'aria-pressed': String(isAct),
        style: { borderLeftColor: arm.col },
        onclick: function () { spec.onSelect(arm.key); }
      }, [
        M.h('strong', { text: arm.title }),
        M.h('span', { text: arm.desc })
      ]);
      grid.appendChild(btn);
    });
  }

  rebuildSubControl();
  renderZooCard();

  M.pointer(stage.cv, {
    down: function (p) {
      var dx = p.x - sphereBounds.cx, dy = p.y - sphereBounds.cy;
      if (dx * dx + dy * dy > Math.pow(sphereBounds.R * 1.45, 2)) return false;
      dragOrigin = { x: p.x, y: p.y, yaw: yaw, pitch: pitch };
      return true;
    },
    drag: function (p) {
      if (!dragOrigin) return;
      yaw = dragOrigin.yaw + (p.x - dragOrigin.x) * 0.012;
      pitch = M.clamp(dragOrigin.pitch + (p.y - dragOrigin.y) * 0.012, -1.1, 1.1);
      drawAll();
    },
    up: function () { dragOrigin = null; },
    hover: function (p) {
      var dx = p.x - sphereBounds.cx, dy = p.y - sphereBounds.cy;
      stage.cv.style.cursor = (dx * dx + dy * dy <= Math.pow(sphereBounds.R * 1.25, 2)) ? 'grab' : '';
    }
  });

  function drawThumb(ctx, kind, idx, x, y, sz, borderCol, s) {
    var img = thumbImgs[kind][idx % thumbImgs[kind].length];
    ctx.save();
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, sz, sz, 4 * s.sc); else ctx.rect(x, y, sz, sz);
    ctx.clip();
    if (img && img.complete && img.naturalWidth > 0) {
      ctx.drawImage(img, x, y, sz, sz);
    } else {
      ctx.fillStyle = M.rgba(borderCol, 0.18);
      ctx.fillRect(x, y, sz, sz);
    }
    ctx.restore();
    ctx.strokeStyle = borderCol;
    ctx.lineWidth = 1.2 * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, sz, sz, 4 * s.sc); else ctx.rect(x, y, sz, sz);
    ctx.stroke();
  }

  function drawStageCard(ctx, x, y, w, h, title, varied, accentCol, s) {
    ctx.fillStyle = varied ? M.rgba(accentCol, 0.06) : C.surface;
    ctx.strokeStyle = varied ? accentCol : C.border;
    ctx.lineWidth = (varied ? 1.65 : 1.05) * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, 5 * s.sc); else ctx.rect(x, y, w, h);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = varied ? accentCol : C.ink;
    ctx.font = M.font(s, 8.3, 700, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(title, x + 8 * s.sc, y + 7 * s.sc);

    M.pill(ctx, s, varied ? 'VARIED' : 'HELD', x + w - 7 * s.sc, y + 12 * s.sc, {
      align: 'right', color: varied ? accentCol : C.muted, size: 7.2
    });
  }

  function drawDietBox(ctx, rx, ry, rw, rh, s) {
    drawStageCard(ctx, rx, ry, rw, rh, '1. SENSORY DIET', varyAxis === 'input', C.orange, s);
    var innerTop = ry + 26 * s.sc, innerH = rh - 34 * s.sc;
    var pad = 8 * s.sc, gw = rw - 2 * pad;

    if (taskMode === 'clip') {
      /* Paired image + caption rows for Dog & Cat */
      var rowH = (innerH - 8 * s.sc) * 0.5;
      [['dogs', C.blue, '"a photo of a dog"'], ['cats', C.orange, '"a photo of a cat"']].forEach(function (item, idx) {
        var py = innerTop + idx * (rowH + 8 * s.sc);
        ctx.fillStyle = M.rgba(item[1], 0.06);
        ctx.strokeStyle = M.rgba(item[1], 0.45);
        ctx.lineWidth = 1.0 * s.sc;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(rx + pad, py, gw, rowH, 4 * s.sc); else ctx.rect(rx + pad, py, gw, rowH);
        ctx.fill(); ctx.stroke();

        var tSz = Math.min(rowH - 12 * s.sc, 44 * s.sc);
        var ty = py + (rowH - tSz) * 0.5;
        drawThumb(ctx, item[0], 0, rx + pad + 7 * s.sc, ty, tSz, item[1], s);
        ctx.fillStyle = C.green; ctx.font = M.font(s, 9.5, 700);
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('\u2194', rx + pad + 7 * s.sc + tSz + 10 * s.sc, py + rowH * 0.5);

        var chipX = rx + pad + 7 * s.sc + tSz + 20 * s.sc;
        var chipW = Math.max(44 * s.sc, rx + pad + gw - chipX - 6 * s.sc);
        ctx.fillStyle = M.rgba(C.green, 0.12);
        ctx.strokeStyle = M.rgba(C.green, 0.65);
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(chipX, py + rowH * 0.5 - 12 * s.sc, chipW, 24 * s.sc, 4 * s.sc);
        else ctx.rect(chipX, py + rowH * 0.5 - 12 * s.sc, chipW, 24 * s.sc);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = C.ink; ctx.font = M.font(s, 7.5, 600);
        ctx.fillText(item[2], chipX + chipW * 0.5, py + rowH * 0.5);
      });
    } else {
      /* Stacked Dog strip + OR/+ pill + Cat strip */
      var sepH = 18 * s.sc;
      var stripH = (innerH - sepH) * 0.5;
      var dogsOn = dietMode !== 'cats', catsOn = dietMode !== 'dogs';

      [['dogs', innerTop, C.blue, dogsOn, 'Dogs'], ['cats', innerTop + stripH + sepH, C.orange, catsOn, 'Cats']].forEach(function (g) {
        ctx.save();
        ctx.fillStyle = g[3] ? M.rgba(g[2], 0.07) : M.rgba(C.border, 0.20);
        ctx.strokeStyle = g[3] ? M.rgba(g[2], 0.72) : C.border;
        ctx.lineWidth = (g[3] ? 1.2 : 0.9) * s.sc;
        if (!g[3]) ctx.setLineDash([3 * s.sc, 2 * s.sc]);
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(rx + pad, g[1], gw, stripH, 4 * s.sc); else ctx.rect(rx + pad, g[1], gw, stripH);
        ctx.fill(); ctx.stroke();
        ctx.setLineDash([]);

        ctx.globalAlpha = g[3] ? 1.0 : 0.28;
        var gap = 5 * s.sc;
        var sz = Math.min(stripH - 12 * s.sc, (gw - 16 * s.sc - 2 * gap) / 3);
        var totW = 3 * sz + 2 * gap;
        var x0 = rx + pad + (gw - totW) * 0.5, y0 = g[1] + (stripH - sz) * 0.5;
        for (var ti = 0; ti < 3; ti++) drawThumb(ctx, g[0], ti, x0 + ti * (sz + gap), y0, sz, g[2], s);
        ctx.restore();
      });

      M.pill(ctx, s, varyAxis === 'input' ? 'OR / +' : '+', rx + rw * 0.5, innerTop + stripH + sepH * 0.5, {
        align: 'center', color: varyAxis === 'input' ? C.orange : C.muted, size: 7.5
      });
    }
  }

  function drawArchBox(ctx, rx, ry, rw, rh, s) {
    drawStageCard(ctx, rx, ry, rw, rh, '2. ARCHITECTURE', varyAxis === 'arch', C.blue, s);
    var innerTop = ry + 26 * s.sc, innerH = rh - 34 * s.sc;
    var pad = 8 * s.sc, gw = rw - 2 * pad;
    var subH = (innerH - 8 * s.sc) * 0.5;
    var cnnOn = varyAxis === 'task' || archMode === 'cnn';
    var vitOn = varyAxis === 'task' || (varyAxis === 'arch' && archMode === 'vit');

    /* Upper sub-card: CNN 2D Convolution */
    var cy0 = innerTop;
    ctx.fillStyle = cnnOn ? M.rgba(C.blue, 0.07) : M.rgba(C.border, 0.16);
    ctx.strokeStyle = cnnOn ? C.blue : C.border;
    ctx.lineWidth = (cnnOn ? 1.4 : 0.9) * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(rx + pad, cy0, gw, subH, 4 * s.sc); else ctx.rect(rx + pad, cy0, gw, subH);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = cnnOn ? C.blue : C.muted;
    ctx.font = M.font(s, 7.8, 600, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('CNN: 2D Convolution', rx + pad + 6 * s.sc, cy0 + 5 * s.sc);

    var cell = Math.min((subH - 23 * s.sc) / 5.1, (gw * 0.42) / 5.1);
    var gx = rx + pad + 10 * s.sc, gy = cy0 + 18 * s.sc;
    for (var r = 0; r < 5; r++) {
      for (var c = 0; c < 5; c++) {
        var inK = r >= 1 && r <= 3 && c >= 1 && c <= 3;
        ctx.fillStyle = inK ? M.rgba(C.blue, cnnOn ? 0.28 : 0.10) : C.codeBg;
        ctx.strokeStyle = inK && cnnOn ? C.blue : C.border;
        ctx.lineWidth = (inK ? 1.0 : 0.6) * s.sc;
        ctx.fillRect(gx + c * cell, gy + r * cell, cell, cell);
        ctx.strokeRect(gx + c * cell, gy + r * cell, cell, cell);
      }
    }
    var fx = gx + 5 * cell + Math.max(16 * s.sc, gw * 0.14), fy = gy + cell;
    for (var r2 = 0; r2 < 3; r2++) {
      for (var c2 = 0; c2 < 3; c2++) {
        var ctr = r2 === 1 && c2 === 1;
        ctx.fillStyle = ctr ? (cnnOn ? C.blue : C.muted) : M.rgba(C.blue, 0.14);
        ctx.strokeStyle = C.border; ctx.lineWidth = 0.6 * s.sc;
        ctx.fillRect(fx + c2 * cell, fy + r2 * cell, cell, cell);
        ctx.strokeRect(fx + c2 * cell, fy + r2 * cell, cell, cell);
      }
    }
    ctx.strokeStyle = M.rgba(C.blue, cnnOn ? 0.65 : 0.25); ctx.lineWidth = 1.0 * s.sc;
    ctx.beginPath();
    ctx.moveTo(gx + 4 * cell, gy + cell); ctx.lineTo(fx + 1.5 * cell, fy + 1.5 * cell);
    ctx.moveTo(gx + 4 * cell, gy + 4 * cell); ctx.lineTo(fx + 1.5 * cell, fy + 1.5 * cell);
    ctx.stroke();

    /* Lower sub-card: ViT Attention Head */
    var vy0 = innerTop + subH + 8 * s.sc;
    ctx.fillStyle = vitOn ? M.rgba(C.green, 0.08) : M.rgba(C.border, 0.16);
    ctx.strokeStyle = vitOn ? C.green : C.border;
    ctx.lineWidth = (vitOn ? 1.4 : 0.9) * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(rx + pad, vy0, gw, subH, 4 * s.sc); else ctx.rect(rx + pad, vy0, gw, subH);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = vitOn ? C.green : C.muted;
    ctx.font = M.font(s, 7.8, 600, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('ViT: Self-Attention', rx + pad + 6 * s.sc, vy0 + 5 * s.sc);

    var xSrc = rx + pad + 18 * s.sc, xDst = rx + pad + gw - 18 * s.sc;
    var yTop = vy0 + 21 * s.sc, stepY = Math.max(8 * s.sc, (subH - 28 * s.sc) / 3);
    var attnW = [[0.5, 0.2, 0.8, 0.1], [0.2, 0.7, 0.2, 0.6], [0.8, 0.2, 0.6, 0.3], [0.1, 0.6, 0.3, 0.7]];
    for (var ai = 0; ai < 4; ai++) {
      for (var aj = 0; aj < 4; aj++) {
        var aw = attnW[ai][aj];
        if (aw < 0.25) continue;
        ctx.strokeStyle = M.rgba(C.green, (vitOn ? 0.72 : 0.22) * aw);
        ctx.lineWidth = (0.6 + aw * 1.6) * s.sc;
        ctx.beginPath();
        ctx.moveTo(xSrc + 4 * s.sc, yTop + ai * stepY);
        ctx.bezierCurveTo((xSrc + xDst) * 0.5, yTop + ai * stepY, (xSrc + xDst) * 0.5, yTop + aj * stepY, xDst - 4 * s.sc, yTop + aj * stepY);
        ctx.stroke();
      }
    }
    for (var ak = 0; ak < 4; ak++) {
      M.dot(ctx, xSrc, yTop + ak * stepY, 3.3 * s.sc, C.surface, vitOn ? C.green : C.muted, 1.2 * s.sc);
      M.dot(ctx, xDst, yTop + ak * stepY, 3.5 * s.sc, vitOn ? C.green : C.muted, C.surface, 0.9 * s.sc);
    }
  }

  function drawSphereBox(ctx, rx, ry, rw, rh, s) {
    drawStageCard(ctx, rx, ry, rw, rh, '3. OBJECTIVE / LATENT HYPERSPHERE', varyAxis === 'task', C.green, s);

    var innerTop = ry + 24 * s.sc, innerBot = ry + rh - 24 * s.sc;
    var cx = rx + rw * 0.5, cy = (innerTop + innerBot) * 0.5 + 2 * s.sc;
    var R = Math.min(rw * 0.39, (innerBot - innerTop) * 0.45);
    sphereBounds = { cx: cx, cy: cy, R: R };

    M.drawSphere(ctx, cx, cy, R, yaw, pitch, s);

    var targ = targetPositions(taskMode);
    var famCols = [C.blue, C.orange];

    if (taskMode === 'supervised') {
      [['Label: "Dog"', SUP_CENTERS[0], C.blue], ['Label: "Cat"', SUP_CENTERS[1], C.orange]].forEach(function (item) {
        var sp = M.sphereProject(item[1], cx, cy, R, yaw, pitch);
        ctx.strokeStyle = M.rgba(item[2], 0.55);
        ctx.setLineDash([3 * s.sc, 2 * s.sc]);
        ctx.lineWidth = 1.2 * s.sc;
        ctx.beginPath(); ctx.arc(sp.x, sp.y, 15 * s.sc, 0, Math.PI * 2); ctx.stroke();
        ctx.setLineDash([]);
        M.star(ctx, sp.x, sp.y, 5.8 * s.sc, C.green, C.surface);
        M.pill(ctx, s, item[0], sp.x, sp.y - 15 * s.sc, { align: 'center', color: item[2], size: 7.6 });
      });
    } else if (taskMode === 'clip') {
      [['"a photo of a dog"', SUP_CENTERS[0], C.blue], ['"a photo of a cat"', SUP_CENTERS[1], C.orange]].forEach(function (item) {
        var sp2 = M.sphereProject(item[1], cx, cy, R, yaw, pitch);
        M.pill(ctx, s, item[0], sp2.x, sp2.y - 16 * s.sc, { align: 'center', color: item[2], size: 7.4 });
      });
    }

    for (var i = 0; i < N; i++) {
      var f = fam[i];
      if (dietMode === 'dogs' && f !== 0) continue;
      if (dietMode === 'cats' && f !== 1) continue;

      var curP = M.normVec([
        M.lerp(initPts[i][0], targ.pts[i][0], prog),
        M.lerp(initPts[i][1], targ.pts[i][1], prog),
        M.lerp(initPts[i][2], targ.pts[i][2], prog)
      ]);
      var curT = M.normVec([
        M.lerp(initTwins[i][0], targ.twins[i][0], prog),
        M.lerp(initTwins[i][1], targ.twins[i][1], prog),
        M.lerp(initTwins[i][2], targ.twins[i][2], prog)
      ]);

      var pp = M.sphereProject(curP, cx, cy, R, yaw, pitch);
      var col = famCols[f];
      var front = pp.z >= 0;

      if (taskMode === 'contrastive' || taskMode === 'clip') {
        var pt = M.sphereProject(curT, cx, cy, R, yaw, pitch);
        ctx.strokeStyle = M.rgba(col, front ? 0.55 : 0.2);
        ctx.lineWidth = 1.15 * s.sc;
        ctx.beginPath(); ctx.moveTo(pp.x, pp.y); ctx.lineTo(pt.x, pt.y); ctx.stroke();
        if (taskMode === 'clip') {
          M.dot(ctx, pt.x, pt.y, 3.2 * s.sc, C.surface, M.rgba(C.green, front ? 0.92 : 0.35), 1.4 * s.sc);
        } else {
          M.dot(ctx, pt.x, pt.y, 2.6 * s.sc, M.rgba(col, front ? 0.75 : 0.25));
        }
      }
      M.dot(ctx, pp.x, pp.y, (front ? 3.7 : 2.6) * s.sc, M.rgba(col, front ? 0.92 : 0.30), front ? C.surface : null, 0.9);
    }

    var caption = {
      supervised: 'Supervised: points cluster at class poles ("Dog", "Cat")',
      autoencode: 'Autoencoder: smooth pose & appearance manifolds',
      contrastive: 'Contrastive: pull augmented views, push instances apart',
      clip: 'CLIP: align image (\u25cf) & text caption (\u25cb) embeddings'
    }[taskMode];
    M.pill(ctx, s, caption, cx, ry + rh - 12 * s.sc, { align: 'center', color: C.ink, size: 7.5 });
  }

  function drawFlowArrow(ctx, x0, y0, x1, y1, s) {
    M.arrow(ctx, x0, y0, x1, y1, C.ink, 2.1 * s.sc, 7.5 * s.sc);
  }

  function drawAll() {
    if (!ready || !stage.w) return;
    var ctx = stage.ctx, s = stage;
    M.clear(ctx, s);
    var pad = 10 * s.sc;

    if (s.w >= 620) {
      /* Single left-to-right pipeline: [1. Sensory Diet] => [2. Architecture] => [3. Objective / Hypersphere] */
      var arrowW = 22 * s.sc;
      var availW = s.w - 2 * pad - 2 * arrowW;
      var w1 = Math.round(availW * 0.27);
      var w2 = Math.round(availW * 0.27);
      var w3 = availW - w1 - w2;
      var hBox = s.h - 2 * pad;

      var x1 = pad;
      var x2 = x1 + w1 + arrowW;
      var x3 = x2 + w2 + arrowW;
      var midY = pad + hBox * 0.52;

      drawDietBox(ctx, x1, pad, w1, hBox, s);
      drawFlowArrow(ctx, x1 + w1 + 3 * s.sc, midY, x2 - 3 * s.sc, midY, s);
      drawArchBox(ctx, x2, pad, w2, hBox, s);
      drawFlowArrow(ctx, x2 + w2 + 3 * s.sc, midY, x3 - 3 * s.sc, midY, s);
      drawSphereBox(ctx, x3, pad, w3, hBox, s);
    } else {
      /* Compact responsive layout inside the single panel on narrow viewports */
      var topH = Math.round((s.h - 2 * pad - 20 * s.sc) * 0.44);
      var botH = s.h - 2 * pad - 20 * s.sc - topH;
      var arrowSm = 16 * s.sc;
      var halfW = (s.w - 2 * pad - arrowSm) * 0.5;
      drawDietBox(ctx, pad, pad, halfW, topH, s);
      drawFlowArrow(ctx, pad + halfW + 2 * s.sc, pad + topH * 0.5, pad + halfW + arrowSm - 2 * s.sc, pad + topH * 0.5, s);
      drawArchBox(ctx, pad + halfW + arrowSm, pad, halfW, topH, s);
      drawFlowArrow(ctx, s.w * 0.5, pad + topH + 2 * s.sc, s.w * 0.5, pad + topH + 18 * s.sc, s);
      drawSphereBox(ctx, pad, pad + topH + 20 * s.sc, s.w - 2 * pad, botH, s);
    }

    var holdSummary = varyAxis === 'task'
      ? 'Sensory Diet (Dogs+Cats) + Architecture (CNN/ViT)'
      : varyAxis === 'arch'
        ? 'Sensory Diet (Dogs+Cats) + Objective (Supervised)'
        : 'Architecture (CNN) + Objective (Contrastive)';
    read.innerHTML = '<span class="k">varied pressure</span> ' + (varyAxis === 'task' ? 'Task Objective (' + taskMode + ')' : varyAxis === 'arch' ? 'Architecture (' + archMode.toUpperCase() + ')' : 'Sensory Diet (' + dietMode + ')') +
      '<span class="sep">&middot;</span><span class="k">held constant</span> ' + holdSummary +
      '<div class="wread-interp"><span class="g">Holding two pressures constant while varying the third turns open-source model zoos into controlled experiments.</span></div>';
  }

  stage.draw = drawAll;
  ready = true;
  tr.show(60);
  stage.request();
  return { destroy: function () { tr.pause(); } };
});
